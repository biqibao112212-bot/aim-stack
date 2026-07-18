#include "timer.h"

namespace rm {

Timer::Timer()
{
    _start_time = _end_time = high_resolution_clock::now();
}

tp Timer::tic()
{
    _start_time = high_resolution_clock::now();
    return _start_time;
}

double Timer::toc(string tag)
{
    _end_time = high_resolution_clock::now();
    double time = getms(_start_time, _end_time);
    if (!tag.empty())
    {
        tag += ": ";
        cout << tag << time << "ms" << endl;
    }
    return time;
}

double Timer::getms(tp start, tp end)
{
    return 1000.0 * duration_cast<std::chrono::duration<double>>(end - start).count();
}

void Timer::waitFor(double second)
{
    tic();
    while(toc() < second * 1000);
}

bool Timer::calTime(double period)
{
    if (period < 0) return true;

    if (!have_timer)
    {
        _timer_start = high_resolution_clock::now();
        have_timer = 1;
    }
    _timer_end = high_resolution_clock::now();
    if (getms(_timer_start, _timer_end) >= period*1000)
    {
        _timer_start = high_resolution_clock::now();
        return true;
    }
    else return false;
}

int Timer::getFreq(string tag)
{
    if (_freq_count==0) _time_used_to_getFreq = high_resolution_clock::now();

    double delta_time = 1000 * std::chrono::duration_cast<std::chrono::duration<double>>
                              (chrono::high_resolution_clock::now()-_time_used_to_getFreq).count();
    if (delta_time >= 1000)
    {
        _time_used_to_getFreq = chrono::high_resolution_clock::now();

        if (!tag.empty())
        {
            tag += ": ";
            cout << tag << _freq_count << endl;
        }
        FPS = _freq_count;
        _freq_count = 0;
    }
    else
    {
        _freq_count++;
    }
    return FPS;
}





}
