pub fn RingBuffer(comptime T: type, comptime N: usize) type {
    return struct {
        data: [N]T = undefined,
        head: usize = 0,
        tail: usize = 0,
        count: usize = 0,
        const Self = @This();
        pub fn init() Self {
            return Self{
                .data = undefined,
                .head = 0,
                .tail = 0,
                .count = 0,
            };
        }

        pub fn push(self: *Self, item: T) bool {
            if (self.count == N) return false;
            self.data[self.tail] = item;
            self.tail = (self.tail + 1) % N;
            self.count += 1;
            return true;
        }

        pub fn pop(self: *Self) ?T {
            if (self.count == 0) return null;
            const item = self.data[self.head];
            self.head = (self.head + 1) % N;
            self.count -= 1;
            return item;
        }

        pub fn is_empty(self: *Self) bool {
            return self.count == 0;
        }
    };
}
