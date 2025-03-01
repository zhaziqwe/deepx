#include "op.hpp"

namespace deepx::op
{
    void Op::load(const char *str)
    {
        // 新格式示例：mul@float32 a(a_grad) b(b_grad) -> a(a_grad) requires_grad=true
        string input(str);
        size_t arrow_pos = input.find("->");
        string head = input.substr(0, arrow_pos);
        string tail = arrow_pos != string::npos ? input.substr(arrow_pos + 2) : "";

        // 解析操作名和类型
        size_t at_pos = head.find('@');
        if (at_pos != string::npos)
        {
            name = head.substr(0, at_pos);
            dtype = head.substr(at_pos + 1, head.find(' ') - at_pos - 1);
            head = head.substr(head.find(' ') + 1);
        }
        else
        {
            name = head.substr(0, head.find(' '));
            dtype = "any";
            head = head.substr(name.size() + 1);
        }

        // 解析输入参数（支持带括号的梯度名）
        stringstream head_ss(head);
        string token;
        while (head_ss >> token)
        {
            size_t bracket = token.find('(');
            if (bracket != string::npos)
            {
                args.push_back(token.substr(0, bracket));
                args_grad.push_back(token.substr(bracket + 1, token.find(')') - bracket - 1));
                require_grad = true;
            }
            else
            {
                args.push_back(token);
            }
        }

        // 解析输出参数和标志
        stringstream tail_ss(tail);
        while (tail_ss >> token)
        {
            if (token.find('(') != string::npos)
            {
                size_t bracket = token.find('(');
                returns.push_back(token.substr(0, bracket));
                returns_grad.push_back(token.substr(bracket + 1, token.find(')') - bracket - 1));
            }
            else if (token == "requires_grad=true")
            {
                require_grad = true;
            }
            else
            {
                returns.push_back(token);
            }
        }
    }

    void Op::init(const string &opname,
                  const string &dtype,
                  const vector<string> &args,
                  const vector<string> &returns,
                  bool require_grad,
                  const vector<string> &args_grad,
                  const vector<string> &returns_grad)
    {
        this->name = opname;
        this->dtype = dtype;
        this->args = args;
        this->returns = returns;
        this->require_grad = require_grad;

        if (require_grad)
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
}