#include <mutex>
#include <thread>
#include <cstdlib>

#include <deepx/tensorfunc/init.hpp>
#include "deepx/tf/tf.hpp"
#include "deepx/tf/tffactory.hpp"
#include "deepx/mem/mem.hpp"
#include "client/udpserver.hpp"

using namespace deepx::tensorfunc;
using namespace deepx::mem;

// 从环境变量读取IR日志配置
bool kIrLog = []()
{
    const char *env = std::getenv("DEEPX_IR_LOG");
    return env != nullptr && (strcmp(env, "1") == 0 || strcasecmp(env, "true") == 0);
}();

int main()
{
    Mem mem;
    std::mutex memmutex;

    client::udpserver server(8080);
    deepx::tf::TfFactory tf_factory;
    register_all(tf_factory);

    tf_factory.print_markdown();

    // 将op table输出到markdown文件
    string docdir = "../../../doc/excuter/op-mem-ompsimd/";
    std::ofstream md_file(docdir + "list.md");
    if (md_file.is_open())
    {
        md_file << tf_factory.print_markdown();
        md_file.close();
    }

    queue<deepx::tf::TF> tasks;
    // 启动一个新线程来运行UDP服务器
    std::thread server_thread([&server, &tasks]()
                              { server.start(tasks); });
    // 分离线程，让它在后台运行
    server_thread.detach();

    while (true)
    {
        if (!tasks.empty())
        {
            deepx::tf::TF op = tasks.front();
            tasks.pop();

            // 根据kIrLog标志决定是否打印op信息
            if (kIrLog)
            {
                cout << "~" << op.to_string() << endl;
            }

            deepx::tf::OpResp opresp;
            opresp.id = op.id;
            opresp.recv_at = op.recv_at;

            if (tf_factory.tf_families.find(op.name) == tf_factory.tf_families.end())
            {
                cerr << "<op> " << op.name << " not found" << endl;
                opresp.error("op" + op.name + " not found");
                continue;
            }
            auto op_family = *(tf_factory.tf_families.find(op.name)->second);
            string author = op.author;
            if (op.author == "")
            {
                author = op_family._default;
                if (author == "" && op_family.tf_authors.size() > 0)
                {
                    author = op_family.tf_authors.begin()->first;
                }
                else
                {
                    cerr << "<op> " << op.name << " no author implement" << endl;
                    opresp.error("op" + op.name + " no author implement");
                    continue;
                }
            }
            if (op_family.tf_authors.find(author) == op_family.tf_authors.end())
            {
                cerr << "<op> " << op.name << " " << author << " not found" << endl;
                opresp.error("op" + op.name + " " + author + " not found");
                continue;
            }
            string dtypes = op.dtypes();
            if (op_family.tf_authors.find(author)->second->tfs.find(dtypes) == op_family.tf_authors.find(author)->second->tfs.end())
            {
                cerr << "<op> " << op.name << " " << author << " " << dtypes << " not found" << endl;
                opresp.error("op" + op.name + " " + author + " " + dtypes + " not found");
                continue;
            }
            auto src = op_family.tf_authors.find(author)->second->tfs.find(dtypes)->second;

            (*src).init(op.name, op.args, op.returns);
            memmutex.lock();
            opresp.start_at = chrono::system_clock::now();

            int ret = (*src).run(mem,opresp.message);
            memmutex.unlock();
            if (ret != 0)
            {
                opresp.error(opresp.message);
                server.resp(opresp.to_string());
                continue;
            }
            opresp.finish("");
            server.resp(opresp.to_string());
        }
    }
    return 0;
}
