#ifndef __SERVER_HPP__
#define __SERVER_HPP__
#include <iostream>
#include <string.h>
#include <arpa/inet.h>
#include <sys/un.h>
#include <unistd.h>
#include <functional>

namespace client
{
    class unixsocketserver
    {
    private:
        std::string socket_path;
        int sockfd;
        struct sockaddr_un servaddr, cliaddr; // 修改为使用完整类型
        char buffer[1024];
        socklen_t len;
        ssize_t n;

    public:
        unixsocketserver(const std::string &path);
        ~unixsocketserver();
        void start();
        using handlefunc = std::function<void(char *buffer)>;
        handlefunc func;
    };
}

#endif