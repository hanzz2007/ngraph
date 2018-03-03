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

#include <stdexcept>

#include "mnist.hpp"

MNistLoader::MNistLoader(const std::string& filename, uint32_t magic)
    : m_filename(filename)
    , m_magic(magic)
{
}

template <>
void MNistLoader::read<std::uint32_t>(std::uint32_t* loc)
{
    std::uint32_t result;
    result = static_cast<unsigned int>(fgetc(m_file)) << 24;
    result |= fgetc(m_file) << 16;
    result |= fgetc(m_file) << 8;
    result |= fgetc(m_file);
    *loc = result;
}

void MNistLoader::read_header()
{
    std::uint32_t m;
    read<std::uint32_t>(&m);
    if (m != m_magic)
    {
        throw std::invalid_argument("Incorrect magic");
    }
    read<std::uint32_t>(&m_items);
}

void MNistLoader::open()
{
    if (m_file != nullptr)
    {
        throw std::logic_error("Loader already open");
    }
    m_file = fopen(m_filename.c_str(), "rb");
    read_header();
    fgetpos(m_file, &m_data_pos);
}

void MNistLoader::reset()
{
    fsetpos(m_file, &m_data_pos);
}

void MNistLoader::close()
{
    if (nullptr != m_file)
    {
        fclose(m_file);
        m_file = nullptr;
    }
}

MNistLoader::~MNistLoader()
{
    close();
}

const char* const MNistImageLoader::TRAIN{"train-images-idx3-ubyte"};
const char* const MNistImageLoader::TEST{"t10k-images-idx3-ubyte"};

MNistImageLoader::MNistImageLoader(const std::string& file_name)
    : MNistLoader(file_name, magic_value)
{
}

void MNistImageLoader::read_header()
{
    MNistLoader::read_header();
    read<uint32_t>(&m_rows);
    read<uint32_t>(&m_columns);
}

const char* MNistLabelLoader::TEST{"t10k-labels-idx1-ubyte"};
const char* MNistLabelLoader::TRAIN{"train-labels-idx1-ubyte"};

MNistLabelLoader::MNistLabelLoader(const std::string& file_name)
    : MNistLoader(file_name, magic_value)
{
}

MNistDataLoader::MNistDataLoader(size_t batch_size,
                                 const std::string& image,
                                 const std::string& label)
    : m_batch_size(batch_size)
    , m_image_loader(image)
    , m_label_loader(label)
{
}

MNistDataLoader::~MNistDataLoader()
{
    close();
}

void MNistDataLoader::open()
{
    m_image_loader.open();
    m_label_loader.open();
    if (m_image_loader.get_items() != m_label_loader.get_items())
    {
        throw std::invalid_argument("Mismatch between image and label items");
    }
    m_items = m_image_loader.get_items();

    m_image_bytes =
        std::unique_ptr<std::uint8_t[]>(new uint8_t[m_batch_size * get_rows() * get_columns()]);
    m_label_bytes = std::unique_ptr<std::uint8_t[]>(new uint8_t[m_batch_size]);
    m_image_floats = std::unique_ptr<float[]>(new float[m_batch_size * get_rows() * get_columns()]);
    m_label_floats = std::unique_ptr<float[]>(new float[m_batch_size * 10]);
}

void MNistDataLoader::close()
{
    m_image_loader.close();
    m_label_loader.close();
}
