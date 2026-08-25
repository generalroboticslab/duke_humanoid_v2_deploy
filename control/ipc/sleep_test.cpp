#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <thread>
#include <cmath>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

void timespec_add_ns(timespec& ts, long ns) {
    ts.tv_nsec += ns;
    if (ts.tv_nsec >= 1'000'000'000L) {
        ts.tv_sec += 1;
        ts.tv_nsec -= 1'000'000'000L;
    }
}

void precise_periodic_sleep(timespec& next, long interval_ns) {
    timespec_add_ns(next, interval_ns);
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);
}

void measure_sleep(const std::string& label, int iterations, auto sleep_func) {
    using namespace std::chrono;
    std::vector<double> durations;

    for (int i = 0; i < iterations; ++i) {
        auto start = high_resolution_clock::now();
        sleep_func();
        auto end = high_resolution_clock::now();
        durations.push_back(duration_cast<microseconds>(end - start).count());
    }

    double sum = 0, sq_sum = 0;
    for (double d : durations) sum += d;
    double mean = sum / iterations;
    for (double d : durations) sq_sum += (d - mean) * (d - mean);
    double stddev = std::sqrt(sq_sum / iterations);

    std::cout << std::left << std::setw(30) << label
              << "Avg: " << std::setw(8) << mean << " µs"
              << "StdDev: " << stddev << " µs\n";
}

int main() {
    const int iterations = 1000;
    const long ns_interval = 5000; // 5 µs

    // Set real-time priority and pin to CPU 1
    sched_param sch = {.sched_priority = 80};
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &sch);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(1, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    measure_sleep("std::this_thread::sleep_for", iterations, [] {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1));
    });

    measure_sleep("nanosleep", iterations, [] {
        timespec ts = {0, 1};
        nanosleep(&ts, nullptr);
    });

    timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);
    measure_sleep("clock_nanosleep (ABS)", iterations, [&next, ns_interval] {
        precise_periodic_sleep(next, ns_interval);
    });
}
