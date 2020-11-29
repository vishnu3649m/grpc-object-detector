#include <iostream>

#include <loguru/loguru.hpp>

int main(int argc, char **argv) {
    loguru::init(argc, argv);
    LOG_F(INFO, "Starting gRPC server...");
    return 0;
}
