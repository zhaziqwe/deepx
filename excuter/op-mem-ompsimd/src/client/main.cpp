#include <mutex>
#include <thread>
#include <cstdlib>

#include <deepx/tensorfunc/init.hpp>
#include "deepx/op/op.hpp"
#include "deepx/op/opfactory.hpp"
#include "deepx/mem/mem.hpp"
#include "client/udpserver.hpp"

using namespace deepx::tensorfunc;
using namespace deepx::mem;

// 从环境变量读取IR日志配置
bool kIrLog = []() {
    const char* env = std::getenv("DEEPX_IR_LOG");
    return env != nullptr && (strcmp(env, "1") == 0 || strcasecmp(env, "true") == 0);
}();

int main()
{
    Mem  mem;
    std::mutex memmutex;

    client::udpserver server(8080);
    deepx::op::OpFactory opfactory;
    register_all(opfactory);
    
    opfactory.print_markdown();

    // 将op table输出到markdown文件
    string docdir="../../../doc/excuter/op-mem-ompsimd/";
    std::ofstream md_file(docdir+"list.md");
    if (md_file.is_open()){
        md_file << opfactory.print_markdown();
        md_file.close();
    }
    

    queue<deepx::op::Op> tasks;
    // 启动一个新线程来运行UDP服务器
    std::thread server_thread([&server, &tasks]() {
        server.start(tasks);
    }); 
    // 分离线程，让它在后台运行
    server_thread.detach();

    while (true) {
        if (!tasks.empty()) {
            deepx::op::Op op = tasks.front();
            tasks.pop();
            
            // 根据kIrLog标志决定是否打印op信息
            if (kIrLog) {
                cout << "~" << op.to_string() << endl;
            }

            deepx::op::OpResp opresp;
            opresp.id = op.id;
            opresp.recv_at = op.recv_at;
            
 
            if ( opfactory.op_families.find(op.name)==opfactory.op_families.end()){
                cerr<<"<op> "<<op.name<<" not found"<<endl;
                opresp.error("op"+op.name+" not found");
                continue;
            } 
            auto op_family = *(opfactory.op_families.find(op.name)->second);
            string op_author_name= op.author;
            if (op.author==""){
                op_author_name= op_family._default;
                if (op_author_name=="" && op_family.op_authors.size()>0){
                    op_author_name=op_family.op_authors.begin()->first;
                }else{
                    cerr<<"<op> "<<op.name<<" no author implement"<<endl;
                    opresp.error("op"+op.name+" no author implement");
                    continue;
                }
            }
            if (op_family.op_authors.find(op_author_name)==op_family.op_authors.end()){
                cerr<<"<op> "<<op.name<<" "<<op_author_name<<" not found"<<endl;
                opresp.error("op"+op.name+" "+op_author_name+" not found");
                continue;
            }
            auto &type_map =*(op_family.op_authors.find(op_author_name)->second);
            if (type_map.ops.find(op.dtype)==type_map.ops.end()){
                cerr<<"<op>"<<op.name<<" "<<op.dtype<<" not found"<<endl;
                 opresp.error("op"+op.dtype+" not found");
                 continue;
            }
            auto src = type_map.ops.find(op.dtype)->second;

            (*src).init(op.name, op.dtype, op.args, op.returns, op.grad, op.args_grad, op.returns_grad);
            memmutex.lock();
            opresp.start_at= chrono::system_clock::now();
            if (op.grad) {
                (*src).backward(mem);
            }else {
            (*src).forward(mem);
            }
            opresp.finish("");
            memmutex.unlock();
            server.resp(opresp.to_string());
        }
    }
    return 0;
}
