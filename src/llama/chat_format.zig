//! Per-model-family chat template helpers for tool calling.
//!
//! Scope: Gemma only for now. Other families (qwen, mistral, llama3) can be
//! added by extending the ModelFamily enum and the switch statements below.
//!
//! Responsibilities:
//!   detect()          — identify model family from the raw Jinja2 template
//!   buildGrammar()    — produce a GBNF string that constrains output to a
//!                       valid tool call JSON with name restricted to declared
//!                       tools; used with llama_sampler_init_grammar_lazy_patterns
//!   triggerPattern()  — return the lazy-grammar trigger string for this family
//!   parseOutput()     — extract name/arguments from the model's raw output

const std = @import("std");

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

pub const ModelFamily = enum { gemma, unknown };

/// Minimal representation of a tool, independent of the API-layer types.
pub const ApiTool = struct {
    name: []const u8,
    description: ?[]const u8 = null,
    /// Raw JSON string of the parameters schema (may be null).
    parameters_json: ?[]const u8 = null,
};

/// Result of parsing the model's raw text output.
pub const ParsedToolCall = struct {
    name: []const u8,
    /// JSON-encoded string of the argument object.
    arguments: []const u8,
    /// Synthetic call ID (e.g. "call_<timestamp>").
    call_id: []const u8,
};

// ---------------------------------------------------------------------------
// Model family detection
// ---------------------------------------------------------------------------

/// Identify the model family from the raw Jinja2 chat template C-string stored
/// in the GGUF metadata.  Pass the value returned by llama_model_chat_template.
pub fn detect(template: ?[*:0]const u8) ModelFamily {
    const tmpl_ptr = template orelse return .unknown;
    const tmpl = std.mem.sliceTo(tmpl_ptr, 0);
    if (std.mem.indexOf(u8, tmpl, "<start_of_turn>") != null) return .gemma;
    return .unknown;
}

// ---------------------------------------------------------------------------
// GBNF grammar builder
// ---------------------------------------------------------------------------

// Static portion of the grammar (value, object, pair, array, string, number,
// ws rules).  Written using Zig's multiline string syntax so backslashes are
// literal — the GBNF parser receives them verbatim, which is what it requires
// for its own escape sequences (e.g. `\\` → literal `\`, `\x7f` → DEL).
const grammar_tail =
    \\value   ::= object | array | string | number | "true" | "false" | "null"
    \\object  ::= "{" ws ( pair ("," ws pair)* )? "}" ws
    \\pair    ::= string ":" ws value
    \\array   ::= "[" ws ( value ("," ws value)* )? "]" ws
    \\string  ::= "\"" ( [^"\\\x7f\x00-\x1f] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* "\"" ws
    \\number  ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [-+]? [0-9]+ )? ws
    \\ws      ::= [ \t\n\r]*
    \\
;

/// Build a null-terminated GBNF grammar string.
///
/// The grammar constrains the region **after** the lazy trigger to:
///   ws {"name": "<one of the declared tool names>", "arguments": <any JSON>}
///
/// Caller owns the returned slice; free with allocator.free().
pub fn buildGrammar(
    allocator: std.mem.Allocator,
    tools: []const ApiTool,
) ![:0]const u8 {
    var buf = std.ArrayList(u8){};

    // root rule — allow optional leading whitespace so newlines between the
    // trigger token and the `{` are accepted.
    //
    // In GBNF, string literals use `\"` to match a literal `"`.  The Zig
    // multiline string passes those characters verbatim to the GBNF parser.
    try buf.appendSlice(allocator,
        \\root    ::= ws "{" ws "\"name\"" ws ":" ws name-val ws "," ws "\"arguments\"" ws ":" ws value ws "}"
        \\
    );

    // name-val rule — one alternative per declared tool.
    // Each alternative is the JSON-encoded string form of the tool name,
    // e.g.  "\"get_weather\""  which GBNF matches against the text  "get_weather".
    try buf.appendSlice(allocator, "name-val ::= ");
    for (tools, 0..) |tool, i| {
        if (i > 0) try buf.appendSlice(allocator, " | ");
        // Build the GBNF string literal  "\"<name>\""
        //   "\"  (Zig produces: quote backslash quote — GBNF open+escaped-quote)
        //   name
        //   \""  (Zig produces: backslash quote quote — GBNF escaped-quote+close)
        try buf.appendSlice(allocator, "\"\\\"");
        try buf.appendSlice(allocator, tool.name);
        try buf.appendSlice(allocator, "\\\"\"");
    }
    try buf.append(allocator, '\n');

    // Append the static value/object/string/... rules.
    try buf.appendSlice(allocator, grammar_tail);

    return buf.toOwnedSliceSentinel(allocator, 0);
}

// ---------------------------------------------------------------------------
// Lazy grammar trigger
// ---------------------------------------------------------------------------

/// The lazy grammar trigger pattern for this model family.
/// The sampler activates the grammar as soon as the model has emitted this
/// string.  Must be a null-terminated C string.
pub fn triggerPattern(family: ModelFamily) [*:0]const u8 {
    return switch (family) {
        .gemma, .unknown => "<tool_call>",
    };
}

// ---------------------------------------------------------------------------
// Output parser
// ---------------------------------------------------------------------------

/// Try to extract a tool call from the model's raw generated text.
///
/// For Gemma (and the generic case) the expected format is:
///   <tool_call>{"name": "fn", "arguments": {...}}</tool_call>
///
/// Returns null if the output does not contain the expected pattern or if the
/// JSON inside is malformed.
pub fn parseOutput(
    allocator: std.mem.Allocator,
    family: ModelFamily,
    content: []const u8,
) ?ParsedToolCall {
    const open_tag: []const u8 = switch (family) {
        .gemma, .unknown => "<tool_call>",
    };

    // Find the opening tag.
    const tag_pos = std.mem.indexOf(u8, content, open_tag) orelse return null;
    const after_tag = content[tag_pos + open_tag.len ..];

    // Strip the closing tag if present.
    const json_str: []const u8 = if (std.mem.indexOf(u8, after_tag, "</tool_call>")) |end|
        after_tag[0..end]
    else
        after_tag;

    const trimmed = std.mem.trim(u8, json_str, " \t\n\r");
    if (trimmed.len == 0 or trimmed[0] != '{') return null;

    // Parse the JSON object.
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, trimmed, .{}) catch return null;
    if (parsed.value != .object) return null;
    const obj = parsed.value.object;

    const name_val = obj.get("name") orelse return null;
    if (name_val != .string) return null;
    const name = name_val.string;

    // Extract arguments as a JSON string.
    const arguments: []const u8 = blk: {
        const av = obj.get("arguments") orelse break :blk "{}";
        switch (av) {
            // OpenAI format: arguments is already a JSON-encoded string.
            .string => |s| break :blk s,
            // Most models emit arguments as a direct JSON object.
            .object, .array => {
                // valueAlloc serialises any zig value to a freshly-allocated
                // JSON string; works with std.json.Value in Zig 0.15.
                break :blk std.json.Stringify.valueAlloc(allocator, av, .{}) catch break :blk "{}";
            },
            else => break :blk "{}",
        }
    };

    const call_id = std.fmt.allocPrint(
        allocator,
        "call_{d}",
        .{std.time.milliTimestamp()},
    ) catch return null;

    return ParsedToolCall{
        .name = name,
        .arguments = arguments,
        .call_id = call_id,
    };
}
