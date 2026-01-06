const enums = @import("enums.zig");
pub const llama = @cImport({
    @cInclude("llama.h");
});
const common_grammar_trigger = struct {
    common_grammar_trigger_type: enums.CommonGrammarTriggerType,
    value: []const u8,
    token: llama.llama_token = llama.LLAMA_TOKEN_NULL,
};
