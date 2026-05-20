const std = @import("std");

// ---------------------------------------------------------------------------
// Request-side tool types
// ---------------------------------------------------------------------------

/// Describes a single function the model can call.
pub const FunctionDefinition = struct {
    name: []const u8,
    description: ?[]const u8 = null,
    /// Raw JSON string of the parameters schema (null if not provided).
    parameters: ?[]const u8 = null,
};

/// A tool entry in the request's "tools" array.
pub const Tool = struct {
    type: []const u8 = "function",
    function: FunctionDefinition,
};

// ---------------------------------------------------------------------------
// Response-side tool call type
// (also appears in conversation history for assistant messages)
// ---------------------------------------------------------------------------

pub const ToolCallFunction = struct {
    name: []const u8,
    /// JSON-encoded string of the argument object, e.g. "{\"city\":\"Paris\"}".
    arguments: []const u8,
};

pub const ToolCall = struct {
    id: []const u8,
    index: i32 = 0,
    type: []const u8 = "function",
    function: ToolCallFunction,
};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

const Error = struct {
    message: []const u8,
    type: []const u8,
    param: ?[]const u8,
    code: ?[]const u8,
};

const ErrorResponse = struct {
    errorResponse: Error,
};

// ---------------------------------------------------------------------------
// Message content types
// ---------------------------------------------------------------------------

pub const ContentPart = union(enum) {
    text: TextContentPart,
    image: ImageContentPart,
    audio: AudioContentPart,
    file: FileContentPart,
};

pub const TextContentPart = struct {
    type: enum { text },
    text: []const u8,
};

pub const ImageContentPart = struct {
    type: enum { image },
    // TODO add url
};

pub const AudioContentPart = struct {
    type: enum { audio },
    // TODO
};

pub const FileContentPart = struct {
    type: enum { file },
    // TODO
};

pub const ContentObject = union(enum) {
    text: []const u8,
    parts: []ContentPart,
};

pub const Content = union(enum) {
    /// Null content — used in assistant messages that only carry tool_calls.
    none,
    plain: []const u8,
    object: ContentObject,
};

// ---------------------------------------------------------------------------
// Message / response types
// ---------------------------------------------------------------------------

pub const Message = struct {
    role: []const u8,
    content: Content,
    /// Present in assistant messages that requested tool calls.
    tool_calls: ?[]ToolCall = null,
    /// Present in tool-role messages; identifies which call this result answers.
    tool_call_id: ?[]const u8 = null,
};

pub const ResponseMessage = struct {
    /// Null when finish_reason is "tool_calls" (only tool_calls is populated).
    content: ?[]const u8,
    tool_calls: ?[]ToolCall = null,
    role: []const u8,
};

/// Delta payload for streaming chunks. All fields are optional so null values
/// are omitted from JSON output when serialized with emit_null_optional_fields=false.
/// OpenAI only sends role in the first delta; subsequent deltas omit it entirely.
pub const StreamDelta = struct {
    role: ?[]const u8 = null,
    content: ?[]const u8 = null,
    tool_calls: ?[]ToolCall = null,
};

pub const Choice = struct {
    finish_reason: ?[]const u8,
    index: i32,
    message: ResponseMessage,
};

pub const ChunkChoice = struct {
    finish_reason: ?[]const u8,
    index: i32,
    delta: StreamDelta,
};

pub const CompleteChunkChoice = struct {
    text: []const u8,
    index: i32,
    finish_reason: ?[]const u8,
};

const Usage = struct {
    completion_tokens: i32 = 0,
    prompt_tokens: i32 = 0,
    total_tokens: i32 = 0,
};

pub const JsonSchema = struct {
    schema: []const u8,
};

pub const ResponseFormat = struct {
    type: []const u8,
    json_schema: ?JsonSchema,
};

const EmbedRequest = struct {
    input: []const u8,
    model: []const u8,
};

pub const StreamOptions = struct {
    include_usage: bool,
};

// ---------------------------------------------------------------------------
// Request / response top-level types
// ---------------------------------------------------------------------------

pub const ChatCompletionRequest = struct {
    model: []const u8,
    messages: []Message,
    stream: ?bool = null,
    stream_options: ?StreamOptions = null,
    max_tokens: ?i32 = null,
    seed: ?i32 = null,
    stop: ?[]?[]const u8 = null,
    temperature: ?f64 = null,
    frequency_penalty: ?f64 = null,
    presence_penalty: ?f64 = null,
    top_p: ?f64 = null,
    response_format: ?ResponseFormat = null,
    tools: ?[]Tool = null,
    /// "auto" | "none" | "required" | {"type":"function","function":{"name":"..."}}
    tool_choice: ?[]const u8 = null,
};

pub const ChatCompletionResponse = struct {
    id: []const u8,
    choices: []Choice,
    created: i64,
    model: []const u8,
    system_fingerprint: []const u8,
    object: []const u8 = "chat.completion",
    usage: Usage,
};

pub const ChatCompletionChunk = struct {
    id: []const u8,
    choices: [1]ChunkChoice,
    created: i64,
    model: []const u8,
    system_fingerprint: []const u8,
    object: []const u8 = "chat.completion.chunk",
    usage: ?Usage = null,
};

pub fn newError(code: i32, message: []const u8) ErrorResponse {
    const etype: []const u8 = switch (code) {
        400 => "invalid_request_error",
        404 => "not_found_error",
        else => "api_error",
    };

    return ErrorResponse{
        .errorResponse = Error{
            .message = message,
            .type = etype,
            .param = null,
            .code = null,
        },
    };
}
