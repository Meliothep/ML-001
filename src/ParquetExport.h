//
// Created by bervo on 19/05/2025.
//

#ifndef PARQUETEXPORT_H
#define PARQUETEXPORT_H

#include "DataExport.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

class ParquetDataExporter : public DataExporter {
public:
    explicit ParquetDataExporter(std::string output_path,  std::shared_ptr<Logger> logger)
        : DataExporter(output_path, logger) {
    }

    int ExportEpochMetrics(std::string output_file, const std::vector<EpochMetrics> &metrics) override {
        arrow::Status status = ExportEpochMetricsParquet(output_file, metrics);
        if (!status.ok()) {
            return 0;
        }
        return 1;
    }

    arrow::Status ExportEpochMetricsParquet(std::string output_file, const std::vector<EpochMetrics> &metrics) const {
        arrow::Int32Builder epoch_builder;
        arrow::FloatBuilder train_loss_builder;
        arrow::FloatBuilder test_loss_builder;
        arrow::FloatBuilder accuracy_builder;

        for (const auto& m : metrics) {
          ARROW_RETURN_NOT_OK(epoch_builder.Append(m.epoch));
          ARROW_RETURN_NOT_OK(train_loss_builder.Append(m.train_loss));
          ARROW_RETURN_NOT_OK(test_loss_builder.Append(m.test_loss));
          ARROW_RETURN_NOT_OK(accuracy_builder.Append(m.accuracy));
        }

        std::shared_ptr<arrow::Array> epoch_arr, train_arr, test_arr, acc_arr;
        ARROW_RETURN_NOT_OK(epoch_builder.Finish(&epoch_arr));
        ARROW_RETURN_NOT_OK(train_loss_builder.Finish(&train_arr));
        ARROW_RETURN_NOT_OK(test_loss_builder.Finish(&test_arr));
        ARROW_RETURN_NOT_OK(accuracy_builder.Finish(&acc_arr));

        auto schema = arrow::schema({
            arrow::field("epoch", arrow::int32()),
            arrow::field("train_loss", arrow::float32()),
            arrow::field("test_loss", arrow::float32()),
            arrow::field("accuracy", arrow::float32())
        });

        auto table = arrow::Table::Make(schema, {epoch_arr, train_arr, test_arr, acc_arr});
        ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(output_path_ + output_file));
        return parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out, 1024);
    };

    int ExportPredictions(std::string output_file, const std::vector<PredictionSample> &metrics) override {
        arrow::Status status = ExportPredictionsParquet(output_file, metrics);
        if (!status.ok()) {
            return 0;
        }
        return 1;
    }

    arrow::Status ExportPredictionsParquet(std::string output_file, const std::vector<PredictionSample> &samples) const {

        arrow::Int32Builder epoch_builder, sample_idx_builder, pred_builder, label_builder;
        arrow::FloatBuilder confidence_builder;
        std::vector<std::shared_ptr<arrow::FloatBuilder>> embedding_builders;

        size_t embedding_dim = samples.empty() ? 0 : samples[0].embedding.size();
        embedding_builders.resize(embedding_dim);
        for (auto& builder : embedding_builders)
            builder = std::make_shared<arrow::FloatBuilder>();

        for (const auto& s : samples) {
            ARROW_RETURN_NOT_OK(epoch_builder.Append(s.epoch));
            ARROW_RETURN_NOT_OK(sample_idx_builder.Append(s.sample_index));
            ARROW_RETURN_NOT_OK(pred_builder.Append(s.predicted_label));
            ARROW_RETURN_NOT_OK(label_builder.Append(s.true_label));
            ARROW_RETURN_NOT_OK(confidence_builder.Append(s.confidence));
            for (size_t i = 0; i < embedding_dim; ++i)
                ARROW_RETURN_NOT_OK(embedding_builders[i]->Append(s.embedding[i]));
        }

        std::vector<std::shared_ptr<arrow::Array>> arrays;
        std::vector<std::shared_ptr<arrow::Field>> fields = {
            arrow::field("epoch", arrow::int32()),
            arrow::field("sample_index", arrow::int32()),
            arrow::field("predicted_label", arrow::int32()),
            arrow::field("true_label", arrow::int32()),
            arrow::field("confidence", arrow::float32())
        };

        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(epoch_builder.Finish(&arr)); arrays.push_back(arr);
        ARROW_RETURN_NOT_OK(sample_idx_builder.Finish(&arr)); arrays.push_back(arr);
        ARROW_RETURN_NOT_OK(pred_builder.Finish(&arr)); arrays.push_back(arr);
        ARROW_RETURN_NOT_OK(label_builder.Finish(&arr)); arrays.push_back(arr);
        ARROW_RETURN_NOT_OK(confidence_builder.Finish(&arr)); arrays.push_back(arr);

        for (size_t i = 0; i < embedding_dim; ++i) {
            ARROW_RETURN_NOT_OK(embedding_builders[i]->Finish(&arr));
            fields.push_back(arrow::field("embed_" + std::to_string(i), arrow::float32()));
            arrays.push_back(arr);
        }

        auto schema = std::make_shared<arrow::Schema>(fields);
        auto table = arrow::Table::Make(schema, arrays);
        //logger_->Log(LogLevel::INFO, output_path_ + output_file);
        ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(output_path_ + output_file));
        return parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out, 1024);
    };
};

#endif //PARQUETEXPORT_H
