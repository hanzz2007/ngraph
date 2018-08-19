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
#include <string>

#include "ngraph/op/util/requires_tensor_view_args.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Custom op
        class Custom : public util::RequiresTensorViewArgs
        {
        public:
            using executor_t =
                std::function<void(const std::vector<std::shared_ptr<runtime::TensorView>>& out,
                                   const std::vector<std::shared_ptr<runtime::TensorView>>& args)>;
            /// \brief Constructs a Custom op.
            ///
            /// \param args Nodes that produces the input tensor.
            Custom(const std::string& name, const NodeVector& args);

            // virtual executor_t get_executor(const std::string& backend_name) = 0;

            virtual void
                execute(runtime::Backend* backend,
                        const std::vector<std::shared_ptr<runtime::TensorView>>& out,
                        const std::vector<std::shared_ptr<runtime::TensorView>>& args) const = 0;

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
        };
    }
}
