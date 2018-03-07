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

#pragma once

#include <memory>
#include <vector>

#include "ngraph/ops/result.hpp"

namespace ngraph
{
    /// \brief Zero or more nodes.
    class ResultVector : public std::vector<std::shared_ptr<op::Result>>
    {
    public:
        ResultVector(size_t size)
            : std::vector<std::shared_ptr<op::Result>>(size)
        {
        }

        ResultVector(const std::initializer_list<std::shared_ptr<op::Result>>& nodes)
            : std::vector<std::shared_ptr<op::Result>>(nodes)
        {
        }

        ResultVector(const std::vector<std::shared_ptr<op::Result>>& nodes)
            : std::vector<std::shared_ptr<op::Result>>(nodes)
        {
        }

        ResultVector(const ResultVector& nodes)
            : std::vector<std::shared_ptr<op::Result>>(nodes)
        {
        }

        ResultVector() {}
    };
}