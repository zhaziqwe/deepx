#include <iostream>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <stdutil/vector.hpp>

#include "deepx/op/op.hpp"


void start_server() {
    int sockfd;
    struct sockaddr_in servaddr, cliaddr;
    char buffer[1024];
    socklen_t len;
    ssize_t n;

    // 创建UDP套接字
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    memset(&servaddr, 0, sizeof(servaddr));
    memset(&cliaddr, 0, sizeof(cliaddr));

    // 绑定IP和端口
    servaddr.sin_family = AF_INET; // IPv4
    servaddr.sin_addr.s_addr = INADDR_ANY;
    servaddr.sin_port = htons(8080);

    if (bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0 ) {
        perror("bind failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    while(true) {
        len = sizeof(cliaddr);  // len is value/result

        // 接收消息
        n = recvfrom(sockfd, (char *)buffer, 1024,
                     MSG_WAITALL, ( struct sockaddr *) &cliaddr,
                     &len);
        buffer[n] = '\0';
        std::cout << "Received message: " << buffer << std::endl;

        try {
            YAML::Node config = YAML::Load(buffer);
            deepx::op::Op op;
            op.load(config);
            std::cout << "Name = " << op.name << ", args = " << op.args << ", returns = " << op.returns  << std::endl;
        } catch (const YAML::Exception& e) {
            std::cerr << "YAML Exception: " << e.what() << std::endl;
        }
    }

    close(sockfd);
}

int main() {
    start_server();
    return 0;
}