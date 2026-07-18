#ifndef TOOLS__THREAD_SAFE_QUEUE_HPP
#define TOOLS__THREAD_SAFE_QUEUE_HPP

#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <atomic>

namespace tools
{
template <typename T, bool PopWhenFull = false>
class ThreadSafeQueue
{
public:
  ThreadSafeQueue(
    size_t max_size, std::function<void(void)> full_handler = [] {})
  : max_size_(max_size), full_handler_(full_handler), running_(true)
  {
  }

  bool push(const T & value)
  {
    std::unique_lock<std::mutex> lock(mutex_);

    if (!running_.load()) {
        return false;
    }

    const bool overwritten = queue_.size() >= max_size_;
    if (overwritten) {
      if (PopWhenFull) {
        queue_.pop();
      } else {
        full_handler_();
        return false;
      }
    }

    queue_.push(value);
    not_empty_condition_.notify_all();
    return overwritten;
  }

  bool push(T && value)
  {
    std::unique_lock<std::mutex> lock(mutex_);

    if (!running_.load()) {
        return false;
    }

    const bool overwritten = queue_.size() >= max_size_;
    if (overwritten) {
      if (PopWhenFull) {
        queue_.pop();
      } else {
        full_handler_();
        return false;
      }
    }

    queue_.push(std::move(value));
    not_empty_condition_.notify_all();
    return overwritten;
  }

  bool wait_pop(T & value)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_condition_.wait(lock, [this] {
      return !queue_.empty() || !running_.load();
    });
    if (queue_.empty()) return false;
    value = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  void pop(T & value)
  {
    std::unique_lock<std::mutex> lock(mutex_);

    not_empty_condition_.wait(lock, [this] { 
        return !queue_.empty() || !running_.load(); 
    });

    if (queue_.empty() && !running_.load()) {
      // (杩斿洖涓€涓粯璁ゅ€硷紝璁╄皟鐢ㄨ€呯煡閬?
      value = T{};
      return;
    }
    
    if (queue_.empty()) {
      std::cerr << "Error: Attempt to pop from an empty queue." << std::endl;
      return;
    }

    value = queue_.front();
    queue_.pop();
  }

  T pop()
  {
    std::unique_lock<std::mutex> lock(mutex_);

   
    // *** 6. 淇敼 wait 鏉′欢 ***
    not_empty_condition_.wait(lock, [this] { 
        return !queue_.empty() || !running_.load(); 
    });
    
    // *** 7. 澧炲姞妫€鏌? 濡傛灉鍥犲仠姝㈣€岃鍞ら啋 ***
    if (queue_.empty() && !running_.load()) {
        // (杩斿洖榛樿鏋勯€犵殑 T锛岃皟鐢ㄨ€呭繀椤绘鏌?
        return T{};
    }
    
    
   
    T value = std::move(queue_.front());
    queue_.pop();
    return std::move(value);
  }

  T front()
  {
    std::unique_lock<std::mutex> lock(mutex_);

    not_empty_condition_.wait(lock, [this] { return !queue_.empty(); });

    return queue_.front();
  }

  void back(T & value)
  {
    std::unique_lock<std::mutex> lock(mutex_);

    if (queue_.empty()) {
      std::cerr << "Error: Attempt to access the back of an empty queue." << std::endl;
      return;
    }

    value = queue_.back();
  }

  bool empty()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  void clear()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
      queue_.pop();
    }
    not_empty_condition_.notify_all();  // 濡傛灉鍏朵粬绾跨▼姝ｅ湪绛夊緟闃熷垪涓嶄负绌猴紝杩欐牱鍙互鍞ら啋瀹冧滑
  }

  void stop()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    running_ = false;
    // 鍞ら啋鎵€鏈夊湪 wait() 涓婄潯鐪犵殑绾跨▼
    not_empty_condition_.notify_all(); 
  }

private:
  std::queue<T> queue_;
  size_t max_size_;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_condition_;
  std::function<void(void)> full_handler_;
  std::atomic<bool> running_;
};

}  // namespace tools

#endif  // TOOLS__THREAD_SAFE_QUEUE_HPP
