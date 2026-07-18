#ifndef TIMER_H
#define TIMER_H

#include <time.h>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace rm
{

using namespace std;
using namespace std::chrono;
using tp = std::chrono::_V2::system_clock::time_point;

class Timer
{
    tp _start_time, _end_time, _time_used_to_getFreq;
    int _freq_count = 0;
    int FPS = 0;
public:
    Timer();

    /**
     * @brief 与toc配合使用
     */
    tp tic();

    /**
     * @brief 用于计时，返回ms
     * 若传入字符串，可直接cout
     * 若不传入，则返回ms和上一个tic之间的ms数
     */
    double toc(string = "");
    double getms(tp start, tp end);

    /**
     * @brief 阻塞定时器
     * 阻塞second秒后继续
     */
    void waitFor(double second);
private:
    tp _timer_start, _timer_end;
    bool have_timer;
public:
    /**
     * @brief 无阻塞定时器
     * 每隔t秒返回一次true，无阻塞
     */
    bool calTime(double t);

    /**
     * @brief 统计帧率Hz
     */
    int getFreq(string = "");
};

}

#endif // TIMER_H
