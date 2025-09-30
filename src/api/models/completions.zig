const std = @import("std");

const finishReasonToolCalls = "tool_calls";

const ToolCall = struct {
    id: []const u8,
    index: i32,
    type: []const u8,
    function: struct {
        name: []const u8,
        arguments: []const u8,
    },
};

const Error = struct {
    message: []const u8,
    type: []const u8,
    param: ?[]const u8,
    code: ?[]const u8,
};

const ErrorResponse = struct {
    errorResponse: Error,
};

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

pub const Content = union(enum) {
    plain: []const u8,
    object: ContentObject,
};

pub const ContentObject = union(enum) {
    text: []const u8,
    parts: []ContentPart,
};

pub const Message = struct {
    role: []const u8,
    content: Content,
    tool_calls: ?[]ToolCall = null,
};

pub const ResponseMessage = struct {
    content: []const u8,
    tool_calls: ?[]ToolCall = null,
    role: []const u8,
};

pub const Choice = struct {
    finish_reason: ?[]const u8,
    index: i32,
    message: ResponseMessage,
};

pub const ChunkChoice = struct {
    finish_reason: ?[]const u8,
    index: i32,
    delta: ResponseMessage,
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
    input: type,
    model: []const u8,
};

pub const StreamOptions = struct {
    include_usage: bool,
};

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
    tools: ?[]ToolCall = null,
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
    object: []const u8 = "chat.completion",
    usage: Usage,
};

pub const CompletionRequest = struct {
    model: []const u8,
    messages: []const u8,
    frequency_penalty: f32,
    max_tokens: ?i32 = null,
    temperature: ?f32 = null,
    presence_penalty: ?f32 = null,
    seed: ?[]const u8,
    stop: ?[]const u8 = null,
    stream: ?bool = null,
    stream_options: ?StreamOptions,
    top_p: ?f32 = null,
    suffix: []const u8,
};

pub const Completion = struct {
    id: []const u8,
    object: []const u8,
    created: i64,
    model: []const u8,
    system_fingerprint: []const u8,
    choices: []CompleteChunkChoice,
    usage: Usage,
};

const CompletionChunk = struct {
    id: []const u8,
    object: []const u8,
    created: i64,
    model: []const u8,
    system_fingerprint: []const u8,
    choices: []CompleteChunkChoice,
    usage: ?*Usage,
};

const Model = struct {
    id: []const u8,
    object: []const u8,
    created: i64,
    owned_by: []const u8,
};

const Embedding = struct {
    object: []const u8,
    embedding: []f32,
    index: i32,
};

const EmbeddingUsage = struct {
    prompt_tokens: i32,
    total_tokens: i32,
};

const EmbeddingList = struct {
    object: []const u8,
    data: []Embedding,
    model: []const u8,
    usage: EmbeddingUsage,
};

const ListCompletion = struct {
    object: []const u8,
    data: []Model,
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
