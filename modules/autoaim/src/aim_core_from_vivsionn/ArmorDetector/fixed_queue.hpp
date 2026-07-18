#ifndef FIXED_QUEUE_HPP
#define FIXED_QUEUE_HPP

#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>

namespace rm {

template <typename T>
class FixedSafeQueue {
public:
    // max_size 默认为 1，保证只有最新帧
    explicit FixedSafeQueue(size_t max_size = 1) : max_size_(max_size) {}

    // 生产者调用：队列满了就丢弃旧的，保证不阻塞也不无限增长
    void push(T new_value) {
        std::lock_guard<std::mutex> lock(mut_);
        while (data_queue_.size() >= max_size_) {
            data_queue_.pop(); // 关键：满了就扔掉旧的
        }
        data_queue_.push(std::move(new_value));
        data_cond_.notify_one();
    }

    void push_back(const T& new_value) {
        push(T(new_value));
    }

    void push_back(T&& new_value) {
        push(std::move(new_value));
    }

    // 消费者调用：如果没数据就等待（避免 CPU 空转），有数据就取走
    bool wait_and_pop(T& value) {
        std::unique_lock<std::mutex> lock(mut_);
        // 等待直到队列不为空
        data_cond_.wait(lock, [this] { return !data_queue_.empty(); });
        value = std::move(data_queue_.front());
        data_queue_.pop();
        return true;
    }

    T pop() {
        T value;
        wait_and_pop(value);
        return value;
    }

    // 非阻塞检查（可选）
    bool empty() const {
        std::lock_guard<std::mutex> lock(mut_);
        return data_queue_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mut_);
        return data_queue_.size();
    }

    // 清空队列
    void clear() {
        std::lock_guard<std::mutex> lock(mut_);
        std::queue<T> empty;
        std::swap(data_queue_, empty);
    }

private:
    mutable std::mutex mut_;
    std::queue<T> data_queue_;
    std::condition_variable data_cond_;
    size_t max_size_;
};

} // namespace rm

#endif // FIXED_QUEUE_HPP
