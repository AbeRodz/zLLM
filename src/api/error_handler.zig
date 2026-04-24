//! OpenAI-compatible error responses, per-request backpressure, and graceful
//! shutdown state — shared across all handler call sites.
//!
//! Backpressure
//! ------------
//! Every inference request (streaming and non-streaming) must call
//! acquireSlot() before touching the llama layer and releaseSlot() exactly
//! once when it finishes.  Requests beyond MAX_CONCURRENT receive a 429
//! immediately, before any GPU work begins.
//!
//! Shutdown
//! --------
//! Call initiateShutdown() from the SIGTERM/SIGINT handler.  After that,
//! acquireSlot() refuses all new requests with ShuttingDown (→ 503).  Call
//! drainRequests() in the main thread to wait for the in-flight counter to
//! reach zero before tearing down the process.

const std = @import("std");
const tk = @import("tokamak");

// ---------------------------------------------------------------------------
// OpenAI wire format
// ---------------------------------------------------------------------------

const OAIErrorBody = struct {
    @"error": struct {
        message: []const u8,
        type: []const u8,
        param: ?[]const u8 = null,
        code: ?[]const u8 = null,
    },
};

const ErrorInfo = struct { status: u16, kind: []const u8, msg: []const u8 };

/// Map any Zig error to an HTTP status code and OpenAI error type string.
pub fn classify(err: anyerror) ErrorInfo {
    return switch (err) {
        error.UnknownModel => .{
            .status = 404,
            .kind = "invalid_request_error",
            .msg = "The model does not exist or is not loaded on this server",
        },

        error.TokenizationFailed,
        error.TemplateFailure,
        error.InvalidContent,
        error.InvalidMessages,
        error.MissingField,
        error.ExpectedObject,
        error.ExpectedArray,
        error.ExpectedString,
        error.InvalidUsage,
        error.DecodeFailed,
        => .{
            .status = 400,
            .kind = "invalid_request_error",
            .msg = "Invalid or malformed request",
        },

        error.TooManyRequests => .{
            .status = 429,
            .kind = "rate_limit_error",
            .msg = "Too many concurrent requests — please retry after a moment",
        },

        error.SessionBusy,
        error.NoSlotAvailable,
        => .{
            .status = 503,
            .kind = "api_error",
            .msg = "Server is overloaded — please retry",
        },

        error.ShuttingDown => .{
            .status = 503,
            .kind = "api_error",
            .msg = "Server is shutting down — please retry on another instance",
        },

        error.InferenceTimeout => .{
            .status = 504,
            .kind = "api_error",
            .msg = "Inference timed out — reduce prompt length or try again",
        },

        else => .{
            .status = 500,
            .kind = "api_error",
            .msg = "Internal server error",
        },
    };
}

/// Write an OpenAI-format JSON error response and flush.
/// Should be called instead of propagating the error to the tokamak framework.
pub fn send(ctx: *tk.Context, err: anyerror) void {
    const info = classify(err);
    std.log.err("handler error: {s} → HTTP {d} ({s})", .{ @errorName(err), info.status, info.kind });
    const body = OAIErrorBody{
        .@"error" = .{ .message = info.msg, .type = info.kind },
    };
    ctx.res.status = info.status;
    ctx.res.content_type = .JSON;
    ctx.res.json(body, .{ .emit_null_optional_fields = false }) catch return;
    ctx.res.write() catch {};
}

// ---------------------------------------------------------------------------
// Backpressure — in-flight request counter
// ---------------------------------------------------------------------------

/// Hard cap on simultaneous active inference requests.
/// Tune this to (GPU VRAM / per-context memory) — the session manager's
/// max_sessions is the long-term cap, this is the concurrent-decode cap.
pub const MAX_CONCURRENT: u32 = 64;

var in_flight = std.atomic.Value(u32).init(0);

/// Try to reserve an inference slot.
///
/// Returns:
///   void             — slot acquired; caller MUST call releaseSlot() exactly once.
///   TooManyRequests  — at capacity; return 429 without touching the GPU.
///   ShuttingDown     — SIGTERM received; return 503.
pub fn acquireSlot() error{ TooManyRequests, ShuttingDown }!void {
    if (shutting_down.load(.acquire)) return error.ShuttingDown;
    var cur = in_flight.load(.acquire);
    while (cur < MAX_CONCURRENT) {
        // cmpxchgWeak returns null on success, the actual value on failure.
        if (in_flight.cmpxchgWeak(cur, cur + 1, .acq_rel, .acquire)) |updated| {
            cur = updated; // lost the race, retry with fresh value
        } else {
            return; // acquired
        }
    }
    return error.TooManyRequests;
}

pub fn releaseSlot() void {
    _ = in_flight.fetchSub(1, .release);
}

pub fn activeRequests() u32 {
    return in_flight.load(.acquire);
}

// ---------------------------------------------------------------------------
// Graceful shutdown
// ---------------------------------------------------------------------------

var shutting_down = std.atomic.Value(bool).init(false);

/// Signal-safe.  Sets the shutdown flag so acquireSlot() starts refusing
/// new work.  Call from SIGTERM / SIGINT handler.
pub fn initiateShutdown() void {
    shutting_down.store(true, .release);
    std.log.info("shutdown: initiated — {d} request(s) still active", .{activeRequests()});
}

pub fn isShuttingDown() bool {
    return shutting_down.load(.acquire);
}

/// Block the calling thread until all in-flight requests complete or
/// timeout_seconds elapses.  Returns true if drained cleanly.
pub fn drainRequests(timeout_seconds: u64) bool {
    const deadline = std.time.nanoTimestamp() +
        @as(i128, @intCast(timeout_seconds)) * std.time.ns_per_s;
    while (in_flight.load(.acquire) > 0) {
        if (std.time.nanoTimestamp() > deadline) return false;
        std.Thread.sleep(10 * std.time.ns_per_ms);
    }
    return true;
}
