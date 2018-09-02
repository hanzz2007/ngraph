/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/op/custom.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"

using namespace std;
using namespace ngraph;

op::Custom::Custom(const string& name, const NodeVector& args)
    : Op(name, args)
{
}

void op::Custom::register_exec(const std::string& backend, execute_t f)
{
    m_execute_collection.insert({backend, f});
}

op::Custom::execute_t op::Custom::get_exec(const std::string& backend_name) const
{
    auto it = m_execute_collection.find(backend_name);
    if (it == m_execute_collection.end())
    {
        throw runtime_error("Custom op '" + get_friendly_name() +
                            "' does not have an execute function for backend '" + backend_name +
                            "'");
    }
    return it->second;
}

int my_plus(int a, int b)
{
    return a + b;
}
int my_minus(int a, int b)
{
    return a - b;
}

void* op::Custom::get_exec_ptr(const std::string& backend_name) const
{
    std::function<int(int, int)> foo = my_plus;
    std::function<int(int, int)> bar = std::plus<int>();

    // calling using functional form:
    std::cout << foo(100, 20) << '\n';
    std::cout << bar(100, 20) << '\n';

    // calling by invoking target:
    std::cout << (*foo.target<int (*)(int, int)>())(100, 20) << '\n';
    std::cout << (*bar.target<std::plus<int>>())(100, 20) << '\n';

    void* f = reinterpret_cast<void*>(*foo.target<int (*)(int, int)>());
    auto g = reinterpret_cast<int (*)(int, int)>(f);
    NGRAPH_INFO;
    NGRAPH_INFO << g(5, 6);
    NGRAPH_INFO;

    // changing target directly:
    *foo.target<int (*)(int, int)>() = &my_minus;
    std::cout << foo(100, 20) << '\n';

    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> args;
    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> out;
    NGRAPH_INFO;
    args.emplace_back(make_shared<runtime::HostTensorView>(ngraph::element::Type(32, 1, 1, "float"),
                                                           ngraph::Shape(2, 2)));
    args.emplace_back(make_shared<runtime::HostTensorView>(ngraph::element::Type(32, 1, 1, "float"),
                                                           ngraph::Shape(2, 2)));
    args.emplace_back(make_shared<runtime::HostTensorView>(ngraph::element::Type(32, 1, 1, "float"),
                                                           ngraph::Shape(2, 2)));
    NGRAPH_INFO;
    out.emplace_back(make_shared<runtime::HostTensorView>(ngraph::element::Type(32, 1, 1, "float"),
                                                          ngraph::Shape(2, 2)));
    NGRAPH_INFO;
    std::cout << __FILE__ << " " << __LINE__ << "\n";

    execute_t exec = get_exec(backend_name);
    void* target = reinterpret_cast<void*>(
        *exec.target<void (*)(
             ngraph::runtime::Backend * backend,
             const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& out,
             const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& args)>());
    if (!target)
    {
        throw runtime_error("Custom op exec function must be static");
    }

    execute_t ABC_3 = *reinterpret_cast<void (*)(
        ngraph::runtime::Backend*,
        const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>&,
        const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>&)>(target);
    NGRAPH_INFO;
    exec(nullptr, out, args);
    NGRAPH_INFO;
    ABC_3(nullptr, out, args);
    NGRAPH_INFO;

    return target;
}

void op::Custom::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
}
