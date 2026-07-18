#ifndef AUTO_BUFF_LATEST_RESULT_MAILBOX_HPP
#define AUTO_BUFF_LATEST_RESULT_MAILBOX_HPP

#include <mutex>
#include <optional>
#include <utility>

namespace auto_buff
{

// Single-consumer, latest-only result handoff. Polling uses try_lock so the
// submit path never waits for either a result or a producer holding the lock.
template <typename Result>
class LatestResultMailbox
{
public:
    void publish(Result result)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_ = std::move(result);
    }

    bool tryPopLatest(Result* result)
    {
        if (result == nullptr) return false;

        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock() || !latest_.has_value()) return false;

        *result = std::move(*latest_);
        latest_.reset();
        return true;
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_.reset();
    }

private:
    std::mutex mutex_;
    std::optional<Result> latest_;
};

}  // namespace auto_buff

#endif  // AUTO_BUFF_LATEST_RESULT_MAILBOX_HPP
