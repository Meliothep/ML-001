#pragma once

#include <string>
#include <vector>
#include <Windows.h>

#include <iostream>
#include <fmt/core.h>

enum class LogLevel { DEBUG = 0, INFO = 1, EXCEPT = 2, CRITICAL = 3};

class LoggerStrategy
{
public:
	LoggerStrategy(LogLevel logLevel);
	virtual ~LoggerStrategy();
	virtual void Log(LogLevel logLevel, std::string message) = 0;

protected:
	CRITICAL_SECTION critical;
	LogLevel m_logLevel;
};

class Logger
{
public:
	Logger();
	virtual ~Logger();

	void AddLoggerStrat(LoggerStrategy* strat);
	void RemoveLoggerStrat(LoggerStrategy* strat);
	void Log(LogLevel logLevel, std::string message);

	template <typename... T>
	void Log(LogLevel logLevel, std::string message, T&&... args) {
		Log(logLevel, fmt::format(message, std::forward<T>(args)...));
	}

protected:
	std::vector<LoggerStrategy*> mp_strategies;
};

class ConsoleLoggerStrategy : public LoggerStrategy
{
public:
	explicit ConsoleLoggerStrategy(LogLevel logLevel);
	void Log(LogLevel logLevel, std::string message) override;
};