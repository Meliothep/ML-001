#ifndef CMAKE_LOGGER_HPP
#define CMAKE_LOGGER_HPP


#pragma once

#include <string>
#include <vector>

enum LogLevel
{
	DEBUG,
	EXCEPT,
	CRITICAL,
	INFO,
};


class LoggerStrategy
{
public:
	LoggerStrategy(LogLevel logLevel);
	virtual ~LoggerStrategy() = default;
	virtual void Log(LogLevel logLevel, std::string message) = 0;
protected:
	LogLevel m_logLevel;
};

class Logger
{
public:
	

	void AddLoggerStrat(LoggerStrategy* strat) {
		mp_strategies.push_back(strat);
	}

	void RemoveLoggerStrat(LoggerStrategy* strat) {
		mp_strategies.erase(strat);
	}

	void Log(LogLevel logLevel, std::string message) {

		if (mp_strategies != nullptr)
		{
			mp_strategies.foreach(Log(logLevel, message));
		}
	};
protected:
	Logger() {};
	virtual ~Logger() {
		delete mp_strategies;
	};

	std::vector<LoggerStrategy*> mp_strategies;
};


class ConsoleLoggerStrategy : public LoggerStrategy
{
public:
	explicit ConsoleLoggerStrategy(LogLevel logLevel)
		: LoggerStrategy(logLevel)
	{
	}

	void Log(LogLevel logLevel, std::string message) override;
};


LoggerStrategy::LoggerStrategy(LogLevel logLevel)
{
	m_logLevel = logLevel;
}

void ConsoleLoggerStrategy::Log(LogLevel logLevel, std::string message)
{
	if (logLevel >= m_logLevel)
	{
		switch (logLevel)
		{
		case LogLevel::DEBUG:
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x02);
			std::cout << "DEBUG : ";
			break;
		case LogLevel::EXCEPT:
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x06);
			std::cout << "EXCEPT : ";
			break;
		case LogLevel::CRITICAL:
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x0C);
			std::cout << "CRITICAL : ";
			break;
		case LogLevel::INFO:
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x01);
			std::cout << "INFO : ";
			break;
		}
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x0F);
		std::cout << message << std::endl;
	}
}
#endif
