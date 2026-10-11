// GrandMassive.cpp : Defines the entry point for the application.
//

#include "GrandMassive.h"
#include "uci.h"

#include <string>

int main(int argc, char* argv[])
{
	uci::Uci engine;

	// Command-line mode: arguments are run as one command, then the program
	// exits. E.g. "GrandMassive bench", "GrandMassive bench 8",
	// "GrandMassive testsuite wac.epd 1000".
	if (argc > 1) {
		std::string cmd = argv[1];
		for (int i = 2; i < argc; ++i) cmd += std::string(" ") + argv[i];
		engine.handle(cmd);
		return 0;
	}

	engine.loop();
	return 0;
}
