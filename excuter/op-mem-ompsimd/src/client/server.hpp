#ifndef __SERVER_HPP__
#define __SERVER_HPP__
#include <iostream>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <functional>
 
namespace client{
    class server
    {
    private:
        int port;
        int sockfd;
        struct sockaddr_in servaddr, cliaddr;
        char buffer[1024];
        socklen_t len;
        ssize_t n;
    public:
        server(int port);
        ~server();
        void start();
        using handlefunc = std::function<void(char *buffer)>;
        handlefunc func;
        
    };
 
}

#endif