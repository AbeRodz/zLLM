//! Per-session llama_context lifecycle manager.
//!
//! Each session owns exactly one llama_context, giving it a fully isolated
//! KV cache. The manager enforces a slot cap and evicts the least-recently-used
//! idle session when the cap is reached.  Expired sessions (idle beyond TTL)
//! are cleaned up lazily on every acquire() call.
//!
//! Threading model
//! ---------------
//! The global mutex protects the session map.  A session's in_use flag is set
//! to true inside acquire() (under the mutex) and cleared inside release().
//! Inference code runs outside the mutex — only map operations are serialised.
//!
//! Lifecycle
//! ---------
//!   globalInit()          — call once at server startup
//!   acquire(id, model)    — borrow a session; creates one if needed
//!   release(session)      — return the session when inference is done
//!
//! Layer-2 note
//! ------------
//! kv_token_count tracks how many tokens are currently in the KV cache.
//! It is always reset to 0 on acquire() for now (full re-encode every request).
//! Layer 2 will use this field to skip re-encoding the unchanged prefix.

const std = @import("std");

const llama_c = @cImport({
    @cInclude("llama.h");
});

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

pub const Session = struct {
    /// Owned copy of the session ID string.
    id: []const u8,
    /// Dedicated llama_context for this session.  Owned by the session;
    /// freed when the session is evicted or the manager is deinitialized.
    ctx: *llama_c.struct_llama_context,
    /// Tokens currently committed to the KV cache (prompt + generated).
    /// Reserved for Layer-2 incremental encoding; always 0 for now.
    kv_token_count: i32 = 0,
    /// Monotonic nanosecond timestamp of the last completed request.
    last_active_ns: i128,
    /// True while a request is actively using this session.
    /// A busy session cannot be acquired or evicted.
    in_use: bool = false,
};

// ---------------------------------------------------------------------------
// SessionManager
// ---------------------------------------------------------------------------

pub const SessionError = error{
    /// The requested session is currently being used by another request.
    SessionBusy,
    /// All slots are occupied by busy sessions; cannot evict any of them.
    NoSlotAvailable,
    /// globalInit() was never called.
    ManagerNotInitialized,
};

pub const SessionManager = struct {
    sessions: std.StringHashMap(*Session),
    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex = .{},
    max_sessions: usize,
    /// Session idle TTL expressed in nanoseconds.
    ttl_ns: i128,
    /// n_ctx used when creating every new llama_context.
    n_ctx_per_session: u32,

    pub fn init(
        allocator: std.mem.Allocator,
        max_sessions: usize,
        ttl_seconds: u64,
        n_ctx: u32,
    ) SessionManager {
        return .{
            .sessions = std.StringHashMap(*Session).init(allocator),
            .allocator = allocator,
            .max_sessions = max_sessions,
            .ttl_ns = @as(i128, @intCast(ttl_seconds)) * std.time.ns_per_s,
            .n_ctx_per_session = n_ctx,
        };
    }

    pub fn deinit(self: *SessionManager) void {
        var it = self.sessions.valueIterator();
        while (it.next()) |ptr| {
            const s = ptr.*;
            llama_c.llama_free(s.ctx);
            self.allocator.free(s.id);
            self.allocator.destroy(s);
        }
        self.sessions.deinit();
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Acquire an idle session by ID.  Creates a new session if none exists.
    ///
    /// The caller MUST call release() once inference is complete.
    ///
    /// On success the KV cache is cleared and kv_token_count reset to 0 so
    /// every request starts from a clean state (Layer-2 will relax this).
    ///
    /// model_opaque — pointer to a llama_model from any @cImport namespace;
    ///               only used when a new context must be created.
    pub fn acquire(
        self: *SessionManager,
        session_id: []const u8,
        model_opaque: *anyopaque,
    ) !*Session {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Lazily reap expired idle sessions before we try to allocate.
        self.evictExpiredLocked();

        // --- Existing session ---
        if (self.sessions.get(session_id)) |existing| {
            if (existing.in_use) return error.SessionBusy;
            existing.in_use = true;
            existing.last_active_ns = std.time.nanoTimestamp();
            // Clear KV so this request starts fresh.
            // Layer 2 will skip this when the prompt prefix is unchanged.
            llama_c.llama_kv_self_clear(existing.ctx);
            existing.kv_token_count = 0;
            return existing;
        }

        // --- New session — make room if at the slot cap ---
        if (self.sessions.count() >= self.max_sessions) {
            if (!self.evictLRULocked()) return error.NoSlotAvailable;
        }

        const model: *llama_c.struct_llama_model = @ptrCast(@alignCast(model_opaque));
        const ctx = try createContext(model, self.n_ctx_per_session);

        const session = try self.allocator.create(Session);
        errdefer {
            llama_c.llama_free(ctx);
            self.allocator.destroy(session);
        }
        const id_copy = try self.allocator.dupe(u8, session_id);
        errdefer self.allocator.free(id_copy);

        session.* = .{
            .id = id_copy,
            .ctx = ctx,
            .kv_token_count = 0,
            .last_active_ns = std.time.nanoTimestamp(),
            .in_use = true,
        };
        try self.sessions.put(id_copy, session);
        std.log.info("session created: id={s} total={d}/{d}", .{
            session_id,
            self.sessions.count(),
            self.max_sessions,
        });
        return session;
    }

    /// Mark the session as idle so it can be acquired by the next request.
    pub fn release(self: *SessionManager, session: *Session) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        session.last_active_ns = std.time.nanoTimestamp();
        session.in_use = false;
        std.log.debug("session released: id={s}", .{session.id});
    }

    // -----------------------------------------------------------------------
    // Internal helpers — all called with self.mutex held
    // -----------------------------------------------------------------------

    /// Evict the least-recently-used idle session.
    /// Returns true if a session was evicted.
    fn evictLRULocked(self: *SessionManager) bool {
        var oldest_ns: i128 = std.math.maxInt(i128);
        var oldest_id: ?[]const u8 = null;

        var it = self.sessions.iterator();
        while (it.next()) |entry| {
            const s = entry.value_ptr.*;
            if (!s.in_use and s.last_active_ns < oldest_ns) {
                oldest_ns = s.last_active_ns;
                oldest_id = entry.key_ptr.*;
            }
        }

        if (oldest_id) |id| {
            if (self.sessions.fetchRemove(id)) |kv| {
                const s = kv.value;
                std.log.info("session evicted (LRU): id={s}", .{s.id});
                llama_c.llama_free(s.ctx);
                self.allocator.free(s.id);
                self.allocator.destroy(s);
                return true;
            }
        }
        return false;
    }

    /// Evict all idle sessions whose idle time exceeds the TTL.
    /// Collects keys on the stack (up to 32) to avoid heap allocation.
    fn evictExpiredLocked(self: *SessionManager) void {
        const now_ns = std.time.nanoTimestamp();
        var expired: [32][]const u8 = undefined;
        var count: usize = 0;

        var it = self.sessions.iterator();
        while (it.next()) |entry| {
            if (count >= expired.len) break;
            const s = entry.value_ptr.*;
            if (!s.in_use and (now_ns - s.last_active_ns) > self.ttl_ns) {
                expired[count] = entry.key_ptr.*;
                count += 1;
            }
        }
        for (expired[0..count]) |id| {
            if (self.sessions.fetchRemove(id)) |kv| {
                const s = kv.value;
                std.log.info("session evicted (TTL): id={s}", .{s.id});
                llama_c.llama_free(s.ctx);
                self.allocator.free(s.id);
                self.allocator.destroy(s);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Context creation
// ---------------------------------------------------------------------------

fn createContext(model: *llama_c.struct_llama_model, n_ctx: u32) !*llama_c.struct_llama_context {
    const cpu = std.Thread.getCpuCount() catch 4;
    // Match llama.zig:llama_context exactly: use all physical cores on Apple
    // Silicon (P+E) for both decode and batch.  The old cpu/4 batch setting
    // was cutting batch throughput to ~25% of available hardware.
    const phys_cpu: usize = blk: {
        if (comptime @import("builtin").os.tag == .macos) {
            var val: c_uint = 0;
            var size: usize = @sizeOf(c_uint);
            if (std.c.sysctlbyname("hw.physicalcpu", &val, &size, null, 0) == 0 and val > 0) {
                break :blk @as(usize, @intCast(val));
            }
        }
        break :blk cpu;
    };
    var params = llama_c.llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = n_ctx;  // must fit any prompt length; llama.cpp chunks internally
    params.n_ubatch = 512;   // physical GPU micro-batch
    params.n_threads = @as(i32, @intCast(phys_cpu));
    params.n_threads_batch = @as(i32, @intCast(phys_cpu));
    const ctx = llama_c.llama_init_from_model(model, params);
    if (ctx == null) return error.FailedToCreateContext;
    return ctx.?;
}

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

var g_manager: ?SessionManager = null;

/// Initialize the global session manager.  Must be called once before the
/// HTTP server starts accepting requests.
pub fn globalInit(
    allocator: std.mem.Allocator,
    max_sessions: usize,
    ttl_seconds: u64,
    n_ctx: u32,
) void {
    g_manager = SessionManager.init(allocator, max_sessions, ttl_seconds, n_ctx);
    std.log.info("SessionManager ready: max_sessions={d} ttl={d}s n_ctx={d}", .{
        max_sessions,
        ttl_seconds,
        n_ctx,
    });
}

/// Acquire a session from the global manager.
pub fn acquire(session_id: []const u8, model_opaque: *anyopaque) !*Session {
    if (g_manager) |*m| return m.acquire(session_id, model_opaque);
    return error.ManagerNotInitialized;
}

/// Release a session back to the global manager.
pub fn release(session: *Session) void {
    if (g_manager) |*m| m.release(session);
}
