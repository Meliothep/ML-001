//
// Created by bervo on 05/05/2025.
//

#ifndef DATAEXPORT_H
#define DATAEXPORT_H

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace dataexport {

  /**
   * A test function that writes sample data to a Parquet file using Arrow C++
   * @param output_path Path where the Parquet file will be written
   * @return arrow::Status indicating success or error
   */
  arrow::Status WriteParquetTest(const std::string& output_path) {
    // Create schema for our data
    std::shared_ptr<arrow::Schema> schema = arrow::schema({
      arrow::field("id", arrow::int32()),
      arrow::field("name", arrow::utf8()),
      arrow::field("value", arrow::float64())
    });

    // Create builders for each column
    arrow::Int32Builder id_builder;
    arrow::StringBuilder name_builder;
    arrow::DoubleBuilder value_builder;

    // Add data to builders
    for (int i = 0; i < 5; ++i) {
      ARROW_RETURN_NOT_OK(id_builder.Append(i));
      ARROW_RETURN_NOT_OK(name_builder.Append("test_" + std::to_string(i)));
      ARROW_RETURN_NOT_OK(value_builder.Append(i * 1.5));
    }

    // Finalize arrays
    std::shared_ptr<arrow::Array> id_array;
    ARROW_RETURN_NOT_OK(id_builder.Finish(&id_array));

    std::shared_ptr<arrow::Array> name_array;
    ARROW_RETURN_NOT_OK(name_builder.Finish(&name_array));

    std::shared_ptr<arrow::Array> value_array;
    ARROW_RETURN_NOT_OK(value_builder.Finish(&value_array));

    // Create table
    std::shared_ptr<arrow::Table> table = arrow::Table::Make(schema, {id_array, name_array, value_array});

    // Open output file
    ARROW_ASSIGN_OR_RAISE(auto outfile, arrow::io::FileOutputStream::Open(output_path));

    // Write to Parquet
    ARROW_RETURN_NOT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, 1024));

    std::cout << "Successfully wrote Parquet file to: " << output_path << std::endl;
    return arrow::Status::OK();
  }

} // namespace dataexport

#endif // DATAEXPORT_H
