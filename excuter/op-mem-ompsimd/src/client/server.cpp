#include "server.hpp"

namespace client
{
    server::server(int port)
    {
        this->port = port;
    };
    server::~server()
    {
        if (sockfd > 0)
        {
            close(sockfd);
        }
    }
    void server::start()
    {
        // 创建UDP套接字
        if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
        {
            perror("socket creation failed");
            exit(EXIT_FAILURE);
        }

        memset(&servaddr, 0, sizeof(servaddr));
        memset(&cliaddr, 0, sizeof(cliaddr));

        // 绑定IP和端口
        servaddr.sin_family = AF_INET; // IPv4
        servaddr.sin_addr.s_addr = INADDR_ANY;
        servaddr.sin_port = htons(port);

        if (bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0)
        {
            perror("bind failed");
            close(sockfd);
            exit(EXIT_FAILURE);
        }
        while (true)
        {
            len = sizeof(cliaddr); // len is value/result
            // 接收消息
            n = recvfrom(sockfd, (char *)buffer, 1024,
                         MSG_WAITALL, (struct sockaddr *)&cliaddr,
                         &len);
            buffer[n] = '\0';
            std::cout << "Received message: " << buffer << std::endl;
            func(buffer);
        }

        close(sockfd);
    }

}