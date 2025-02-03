#include <functional>

#include "deepx/shape_combination.hpp"

namespace deepx
{
    std::vector<std::vector<int>> combination(int n, int k)
    {
        if (k > n || k < 0)
        {
            return {};
        }
        if (k == 0)
        {
            return {{}};
        }

        std::vector<std::vector<int>> result;
        std::vector<int> path;

        // 递归函数
        std::function<void(int)> backtrack = [&](int start)
        {
            if (path.size() == k)
            {
                result.push_back(path);
                return;
            }
            for (int i = start; i < n; i++)
            {
                path.push_back(i);
                backtrack(i + 1);
                path.pop_back(); // 回溯
            }
        };

        backtrack(0);
        return result;
    }
    std::vector<std::vector<int>> combination(int n )
    {
        std::vector<std::vector<int>> result;
        for (int k = 0; k <= n; k++)
        {
            std::vector<std::vector<int>> temp = combination(n, k);
            result.insert(result.end(), temp.begin(), temp.end());
        }
        return result;
    }
}