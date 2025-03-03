
#include <iomanip>
#include <sstream>
#include <ctime>

#include "op.hpp"
namespace deepx::op
{
    // 与deepx/front/py/deepx/nn/deepxir.py对应

    // 新格式示例：mul@float32 a(a_grad) b(b_grad) -> a(a_grad) //id=1 create_time=1714512000 send_time=1714512000 recv_time=1714512000
    void Op::load(const string &input)
    {
        // 分割元数据部分
        size_t meta_pos = input.find("//");
        string body = input.substr(0, meta_pos);
        string meta = (meta_pos != string::npos) ? input.substr(meta_pos + 2) : "";

        // 解析操作主体
        size_t arrow_pos = body.find("->");
        if (arrow_pos == string::npos)
        {
            arrow_pos = body.find("<-");
            if (arrow_pos != string::npos)
            {
                grad = true; // 反向传播标记
            }
        }

        if (arrow_pos == string::npos)
        {
            throw runtime_error("Invalid IR format: missing arrow");
        }

        string head = body.substr(0, arrow_pos);
        string tail = body.substr(arrow_pos + 2);

        // 解析操作名和数据类型
        size_t at_pos = head.find('@');
        if (at_pos != string::npos)
        {
            name = head.substr(0, at_pos);
            size_t space_pos = head.find(' ', at_pos);
            if (space_pos != string::npos)
            {
                dtype = head.substr(at_pos + 1, space_pos - at_pos - 1);
                head = head.substr(space_pos + 1);
            }
            else
            {
                dtype = head.substr(at_pos + 1);
                head.clear();
            }
        }
        else
        {
            size_t space_pos = head.find(' ');
            if (space_pos != string::npos)
            {
                name = head.substr(0, space_pos);
                head = head.substr(space_pos + 1);
                dtype = "any";
            }
            else
            {
                name = head;
                head.clear();
                dtype = "any";
            }
        }

        // 解析输入参数
        stringstream head_ss(head);
        string token;
        while (head_ss >> token)
        {
            size_t bracket = token.find('(');
            if (bracket != string::npos && token.back() == ')')
            {
                args.push_back(token.substr(0, bracket));
                args_grad.push_back(token.substr(bracket + 1, token.size() - bracket - 2));
            }
            else
            {
                args.push_back(token);
                args_grad.emplace_back(""); // 保持梯度与参数数量一致
            }
        }

        // 解析输出参数
        stringstream tail_ss(tail);
        while (tail_ss >> token)
        {
            size_t bracket = token.find('(');
            if (bracket != string::npos && token.back() == ')')
            {
                returns.push_back(token.substr(0, bracket));
                returns_grad.push_back(token.substr(bracket + 1, token.size() - bracket - 2));
            }
            else
            {
                returns.push_back(token);
                returns_grad.emplace_back(""); // 保持梯度与参数数量一致
            }
        }

        // 解析元数据
        if (!meta.empty())
        {
            stringstream meta_ss(meta);
            string key, value;
            while (meta_ss >> key)
            {
                size_t eq_pos = key.find('=');
                if (eq_pos != string::npos)
                {
                    value = key.substr(eq_pos + 1);
                    key = key.substr(0, eq_pos);

                    if (key == "id")
                    {
                        id = stoi(value);
                    }
                    else if (key == "created_at")
                    {
                        created_at = system_clock::from_time_t(stod(value));
                    }
                    else if (key == "sent_at")
                    {
                        sent_at = system_clock::from_time_t(stod(value));
                    }
                }
            }
        }
    }

    void Op::init(const string &opname,
                  const string &dtype,
                  const vector<string> &args,
                  const vector<string> &returns,
                  bool grad,
                  const vector<string> &args_grad,
                  const vector<string> &returns_grad)
    {
        this->name = opname;
        this->dtype = dtype;
        this->args = args;
        this->returns = returns;
        this->grad = grad;

        if (grad)
        {
            // 如果提供了梯度变量名,就使用提供的名字
            if (!args_grad.empty())
            {
                this->args_grad = args_grad;
            }
            // 否则为每个参数添加.grad后缀
            else
            {
                this->args_grad.clear();
                for (const auto &arg : args)
                {
                    this->args_grad.push_back(arg + ".grad");
                }
            }

            // 同样处理返回值的梯度
            if (!returns_grad.empty())
            {
                this->returns_grad = returns_grad;
            }
            else
            {
                this->returns_grad.clear();
                for (const auto &ret : returns)
                {
                    this->returns_grad.push_back(ret + ".grad");
                }
            }
        }
    }
    static std::string format_time(const system_clock::time_point &tp)
    {
        using namespace std::chrono;
        auto ms = duration_cast<microseconds>(tp.time_since_epoch());
        auto sec = duration_cast<seconds>(ms);
        ms -= sec;

        std::time_t t = sec.count();
        std::tm tm;
        localtime_r(&t, &tm); // 线程安全版本

        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S")
            << '.' << std::setfill('0') << std::setw(6) << ms.count();
        return oss.str();
    }
    std::string Op::to_string(bool show_extra) const
    {
        std::stringstream ss;
        ss << name << "@" << dtype;
        for (size_t i = 0; i < args.size(); ++i)
        {
            if (grad)
            {
                ss << " " << args[i] << "(:+)" << args_grad[i];
            }
            else
            {
                ss << " " << args[i];
            }
        }
        ss << " ->";
        for (size_t i = 0; i < returns.size(); ++i)
        {
            if (grad)
            {
                ss << " " << returns[i] << "(:+)" << returns_grad[i];
            }
            else
            {
                ss << " " << returns[i];
            }
        }
        if (show_extra)
        {
            ss << "//id=" << id
               << " created_at=" << format_time(created_at)
               << " sent_at=" << format_time(sent_at)
               << " recv_at=" << format_time(recv_at);
        }
        return ss.str();
    }
}