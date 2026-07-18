#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <opencv2/opencv.hpp>
#include <chrono>
#include <mutex>
#include <memory>
#include <fstream>
#include <ctime>
#include <thread>
#include <condition_variable>

namespace rm
{

template<typename T>
class ConcurrentQueue
{
public:
    ConcurrentQueue(){};
    ~ConcurrentQueue(){};

    bool is_shutdown()
    {
        return this->_shutdown;
    }

    void shut_down()
    {
        std::unique_lock<std::mutex> mlock(mutex_);
        queue_.clear();
        this->_shutdown = true;
    }

    T pop()
    {
        std::unique_lock<std::mutex> mlock(mutex_);
        /*
        while (queue_.empty())
        {
            cond_.wait(mlock);
        }
        */
        cond_.wait(mlock, [this](){ return !this->queue_.empty();});
        T rc(std::move(queue_.front()));
        queue_.pop_front();
        return rc;
    }

    void pop_front()
    {
        std::unique_lock<std::mutex> mlock(mutex_);
        while (queue_.empty())
        {
            cond_.wait(mlock);
        }
        queue_.pop_front();
    }

    void push_back(T &item)
    {
        std::unique_lock<std::mutex> mlock(mutex_);
        queue_.push_back(item);
        if (queue_.size() > 6)
        {
            cond_.wait(mlock, [this](){ return !this->queue_.empty();});
            queue_.pop_front();
    //        printf("队满！\n");
        }
        mlock.unlock();     // unlock before notificiation to minimize mutex con
        cond_.notify_one(); // notify one waiting thread
    }

    void push_back(T &&item)
    {
        std::unique_lock<std::mutex> mlock(mutex_);
        queue_.push_back(std::move(item));
        if (queue_.size() > 106)
        {
            cond_.wait(mlock, [this](){ return !this->queue_.empty();});
            queue_.pop_front();
        }
        mlock.unlock();     // unlock before notificiation to minimize mutex con
        cond_.notify_one(); // notify one waiting thread
    }

    int size()
    {
        std::unique_lock<std::mutex> mlock(mutex_);
        int size = queue_.size();
        mlock.unlock();
        return size;
    }

    bool empty()
    {
        std::unique_lock<std::mutex> mlock(mutex_);
        bool is_empty = queue_.empty();
        mlock.unlock();
        return is_empty;
    }

    void clear()
    {
        std::unique_lock<std::mutex> mlock(mutex_);
        queue_.clear();
    }

private:
    std::deque<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    bool _shutdown = false;
};

}

#endif // FRAMEBUFFER_H
