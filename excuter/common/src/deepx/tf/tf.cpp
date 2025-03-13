#include <iomanip>
#include <sstream>
#include <ctime>

#include "deepx/tf/tf.hpp"
#include "stdutil/time.hpp"
namespace deepx::tf
{
    // 示例：mul(float16 a ,float16 b)->(float16 c)   //author=miaobyte id=1 create_time=1714512000 send_time=1714512000
    // 示例：newtensor(int32 3,4,5)->(float16 A)   //author=miaobyte id=1 create_time=1714512000 send_time=1714512000
    // 解析元数据部分
    std::pair<string, string> split_metadata(const string &input) {
        size_t meta_pos = input.find("//");
        string body = input.substr(0, meta_pos);
        string meta = (meta_pos != string::npos) ? input.substr(meta_pos + 2) : "";
        return {body, meta};
    }

    // 解析函数名
    string parse_function_name(const string &body) {
        size_t paren_start = body.find('(');
        if (paren_start == string::npos) {
            throw runtime_error("Invalid IR format: missing opening parenthesis");
        }
        return body.substr(0, paren_start);
    }

    // 解析数组内容
    vector<string> parse_array_elements(const string &array_str) {
        vector<string> elements;
        stringstream ss(array_str);
        string element;
        
        while (ss >> element) {
            elements.push_back(element);
        }
        
        return elements;
    }

    // 解析单个参数
    Param parse_single_param(const string &param_str, bool call) {
        stringstream param_ss(param_str);
        string first_token, second_token;
        
        if (!(param_ss >> first_token)) {
            return Param();
        }
        
        // 检查是否是数组格式
        if (first_token.front() == '[') {
            if (first_token.back() != ']') {
                // 如果不是单个token的数组，需要读取到']'为止
                string array_content = first_token.substr(1);  // 去掉开头的'['
                string temp;
                while (param_ss >> temp) {
                    array_content += " " + temp;
                    if (temp.back() == ']') {
                        array_content = array_content.substr(0, array_content.length() - 1);  // 去掉结尾的']'
                        if (call) {
                            // callfunc模式：作为数组值
                            return Param("[" + array_content + "]", "any");
                        } else {
                            // deffunc模式：第二个token是参数名
                            string param_name;
                            param_ss >> param_name;
                            return Param(param_name, array_content);
                        }
                    }
                }
                throw runtime_error("Invalid array format: missing closing bracket");
            } else {
                // 单个token的数组
                string array_content = first_token.substr(1, first_token.length() - 2);
                if (call) {
                    // callfunc模式：作为数组值
                    return Param(first_token, "any");
                } else {
                    // deffunc模式：第二个token是参数名
                    string param_name;
                    param_ss >> param_name;
                    return Param(param_name, array_content);
                }
            }
        }
        
        // 非数组格式的处理
        if (param_ss >> second_token) {
            // 有两个token，第一个是类型，第二个是参数名
            string current_type = first_token;
            Param main_param(second_token, current_type);
            return main_param;
        } else {
            // 只有一个token
            if (call) {
                // callfunc模式：token作为参数名/值
                return Param(first_token, "any");
            } else {
                // deffunc模式：token作为类型
                return Param("", first_token);
            }
        }
    }

    // 解析参数列表
    vector<Param> parse_params(const string &params_str, bool call) {
        vector<Param> params;
        stringstream params_ss(params_str);
        string param;
        
        while (getline(params_ss, param, ',')) {
            // 去除首尾空格
            param.erase(0, param.find_first_not_of(" "));
            param.erase(param.find_last_not_of(" ") + 1);
            
            if (param.empty()) continue;
            
            // 解析单个参数
            Param parsed_param = parse_single_param(param, call);
            params.push_back(parsed_param);
        }
        
        return params;
    }

    // 解析元数据键值对
    void parse_metadata_pair(const string &key_value, int &id, string &author, 
                            system_clock::time_point &created_at, 
                            system_clock::time_point &sent_at) {
        size_t eq_pos = key_value.find('=');
        if (eq_pos == string::npos) return;
        
        string key = key_value.substr(0, eq_pos);
        string value = key_value.substr(eq_pos + 1);
        
        if (key == "id") {
            id = stoi(value);
        } else if (key == "author") {
            author = value;
        } else if (key == "created_at") {
            created_at = system_clock::from_time_t(stod(value));
        } else if (key == "sent_at") {
            sent_at = system_clock::from_time_t(stod(value));
        }
    }

    // 解析元数据
    void parse_metadata(const string &meta, int &id, string &author,
                       system_clock::time_point &created_at,
                       system_clock::time_point &sent_at) {
        if (meta.empty()) return;
        
        stringstream meta_ss(meta);
        string key_value;
        while (meta_ss >> key_value) {
            parse_metadata_pair(key_value, id, author, created_at, sent_at);
        }
    }

    // 主解析函数
    void TF::parse(const string &input, bool call ) {
        // 分割元数据
        auto [body, meta] = split_metadata(input);
        
        // 解析操作主体
        size_t arrow_pos = body.find("->");
        if (arrow_pos == string::npos) {
            throw runtime_error("Invalid IR format: missing arrow");
        }
        
        // 获取函数名
        name = parse_function_name(body);
        
        // 解析参数部分
        size_t paren_start = body.find('(');
        size_t paren_end = body.find(')');
        if (paren_start == string::npos || paren_end == string::npos) {
            throw runtime_error("Invalid IR format: missing parentheses");
        }
        
        // 解析输入参数
        string params = body.substr(paren_start + 1, paren_end - paren_start - 1);
        args = parse_params(params, call);
        
        // 验证参数格式
        if (call) {
            // callfunc模式：检查所有参数是否都有名称/值
            for (const auto& arg : args) {
                if (arg.name.empty()) {
                    throw runtime_error("Invalid call format: parameter must have name/value");
                }
            }
        } else {
            // deffunc模式：检查所有参数是否都有类型
            for (const auto& arg : args) {
                if (arg.dtype == "any") {
                    throw runtime_error("Invalid def format: parameter must have type");
                }
            }
        }
        
        // 解析返回值
        string returns_str = body.substr(arrow_pos + 2);
        size_t ret_paren_start = returns_str.find('(');
        size_t ret_paren_end = returns_str.find(')');
        
        if (ret_paren_start == string::npos || ret_paren_end == string::npos) {
            throw runtime_error("Invalid IR format: missing return parentheses");
        }
        
        string ret_params = returns_str.substr(ret_paren_start + 1, ret_paren_end - ret_paren_start - 1);
        returns = parse_params(ret_params, call);
        
        // 验证返回值格式
        if (call) {
            // callfunc模式：检查所有返回值是否都有名称
            for (const auto& ret : returns) {
                if (ret.name.empty()) {
                    throw runtime_error("Invalid call format: return must have name");
                }
            }
        } else {
            // deffunc模式：检查所有返回值是否都有类型
            for (const auto& ret : returns) {
                if (ret.dtype == "any") {
                    throw runtime_error("Invalid def format: return must have type");
                }
            }
        }
        
        // 解析元数据
        parse_metadata(meta, id, author, created_at, sent_at);
    }

    void TF::init(const string &opname,
                  const vector<Param> &args,
                  const vector<Param> &returns)
    {
        this->name = opname;
        this->author = "";
        this->args = args;
        this->returns = returns;
        
    }

    std::string TF::to_string(bool show_extra, bool show_name) const
    {
        std::stringstream ss;
        ss << name << "(";

        // 处理输入参数
        for (size_t i = 0; i < args.size(); ++i)
        {
            if (i > 0)
            {
                ss << ", "; // 始终使用逗号分隔参数
            }

            // 输出类型，根据show_name决定是否输出参数名
            ss << args[i].dtype;
            if (show_name)
            {
                ss << " " << args[i].name;
            }
        }

        ss << ")->(";

        // 处理返回值
        for (size_t i = 0; i < returns.size(); ++i)
        {
            if (i > 0)
            {
                ss << ", "; // 始终使用逗号分隔返回值
            }

            // 输出类型，根据show_name决定是否输出返回值名
            ss << returns[i].dtype;
            if (show_name)
            {
                ss << " " << returns[i].name;
            }
        }

        ss << ")";

        if (show_extra)
        {
            ss << " //id=" << id
               << " created_at=" << stdutil::format_time(created_at)
               << " sent_at=" << stdutil::format_time(sent_at);
        }

        return ss.str();
    }
 
    TF::TF(string text, bool call)
    {
        parse(text,call);
    }
    string TF::op_name()
    {
        return name;
    }

    string TF::math_formula() const
    {
        return "";
    }

    void TF::setexample()
    {
    }
 
    std::string TF::dtypes() const
    {
        std::string full = to_string(false, false);
        size_t start = full.find('(');
        if (start != std::string::npos)
        {
            return full.substr(start);
        }
        return full;
    }

    bool TF::check_dtype(const TF &other) const
    {
        // 检查参数数量是否匹配
        if (args.size() != other.args.size() || returns.size() != other.returns.size())
        {
            return false;
        }

        // 检查每个参数的类型
        for (size_t i = 0; i < args.size(); ++i)
        {
            // 当前TF的类型可能包含多个选项
            std::string dtype = args[i].dtype;
            std::string other_dtype = other.args[i].dtype;

            // 如果当前类型包含多个选项
            if (dtype.find('|') != std::string::npos)
            {
                bool match = false;
                std::stringstream ss(dtype);
                std::string type;

                // 遍历所有可能的类型选项
                while (getline(ss, type, '|'))
                {
                    if (type == other_dtype)
                    {
                        match = true;
                        break;
                    }
                }

                if (!match)
                    return false;
            }
            else
            {
                // 如果是单一类型，直接比较
                if (dtype != other_dtype)
                    return false;
            }
        }

        // 检查每个返回值的类型
        for (size_t i = 0; i < returns.size(); ++i)
        {
            std::string dtype = returns[i].dtype;
            std::string other_dtype = other.returns[i].dtype;

            // 如果当前类型包含多个选项
            if (dtype.find('|') != std::string::npos)
            {
                bool match = false;
                std::stringstream ss(dtype);
                std::string type;

                // 遍历所有可能的类型选项
                while (getline(ss, type, '|'))
                {
                    if (type == other_dtype)
                    {
                        match = true;
                        break;
                    }
                }

                if (!match)
                    return false;
            }
            else
            {
                // 如果是单一类型，直接比较
                if (dtype != other_dtype)
                    return false;
            }
        }

        return true;
    }
}