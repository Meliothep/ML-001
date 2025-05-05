#include "Logger.h"
#include <algorithm>
#include <ctime>

// LoggerStrategy implementation
LoggerStrategy::LoggerStrategy(LogLevel logLevel) {
    m_logLevel = logLevel;
    InitializeCriticalSection(&critical);
}

LoggerStrategy::~LoggerStrategy() {
    DeleteCriticalSection(&critical);
}

// Logger implementation
Logger::Logger() {}

Logger::~Logger() {
    for (auto strategy : mp_strategies) {
        delete strategy;
    }
    mp_strategies.clear();
}

void Logger::AddLoggerStrat(LoggerStrategy* strat) {
    mp_strategies.push_back(strat);
}

void Logger::RemoveLoggerStrat(LoggerStrategy* strat) {
    mp_strategies.erase(std::remove(mp_strategies.begin(), mp_strategies.end(), strat), mp_strategies.end());
}

void Logger::Log(LogLevel logLevel, std::string message) {
    if (mp_strategies.size() > 0) {
        for (LoggerStrategy* strat : mp_strategies) {
            strat->Log(logLevel, message);
        }
    }
}

// ConsoleLoggerStrategy implementation
ConsoleLoggerStrategy::ConsoleLoggerStrategy(LogLevel logLevel)
    : LoggerStrategy(logLevel) {
}

void ConsoleLoggerStrategy::Log(LogLevel logLevel, std::string message) {
    if (logLevel >= m_logLevel) {
        EnterCriticalSection(&critical);

        time_t now = time(0);
        tm* timeinfo = new tm();
        localtime_s(timeinfo, &now);
        char timestamp[20];
        strftime(timestamp, sizeof(timestamp),
            "%Y-%m-%d %H:%M:%S", timeinfo);

        std::cout << "[" << timestamp << "] ";

        switch (logLevel) {
        case LogLevel::DEBUG:
            SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x02);
            std::cout << "DEBUG ";
            break;
        case LogLevel::EXCEPT:
            SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x06);
            std::cout << "EXCEPT ";
            break;
        case LogLevel::CRITICAL:
            SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x0C);
            std::cout << "CRITICAL ";
            break;
        case LogLevel::INFO:
            SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x01);
            std::cout << "INFO ";
            break;
        }
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x0F);
        std::cout << " : " << message << std::endl;

        delete timeinfo; // Fixed memory leak
        LeaveCriticalSection(&critical);
    }
}
