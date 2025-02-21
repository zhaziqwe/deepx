#include <sstream>
 
#include "udpserver.hpp"

namespace client
{
    using namespace std;
    udpserver::udpserver(int port)
    {
        this->port = port;
    };
    udpserver::~udpserver()
    {
        if (sockfd > 0)
        {
            close(sockfd);
        }
    }
    void udpserver::start()
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
            len = sizeof(cliaddr);
            n = recvfrom(sockfd, (char *)buffer, 1024, 0, (struct sockaddr *)&cliaddr, &len);
            buffer[n] = '\0';
            
            // 新增换行拆分逻辑
            stringstream ss(buffer);
            string line;
            while (getline(ss, line)) {
                if (!line.empty()) {
                    cout << "~" << line << endl;
                    char *IR = const_cast<char *>(line.c_str());
                    func(IR);
                }
            }
        }
        close(sockfd);
    }
}