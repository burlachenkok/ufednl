#pragma once

#include "dopt/fs/include/FileSystemHelpers.h"
#include "dopt/fs/include/StringUtils.h"
#include "dopt/random/include/RandomGenRealLinear.h"
#include "dopt/copylocal/include/MutableData.h"
#include "dopt/copylocal/include/Data.h"
#include "dopt/copylocal/include/Copier.h"
#include "dopt/math_routines/include/SimpleMathRoutines.h"

#include <stdio.h>
#include <stddef.h>
#include <math.h>

#include <set>
#include <unordered_set>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <type_traits>

namespace dopt
{
    enum CreateDatasetFlags
    {
        eNone = 0x0,                          ///< None flags
        eAddInterceptTerm = 0x1 << 0,         ///< Add intercept term flag
        eMakeRemappingForBinaryLogisticLoss   ///< Flag which specify that we should perform remapping for binary classification loss
    };

    template <class IndexType, class LabelType>
    struct DataSetLoaderStats
    {
        using TIndexType = IndexType;
        using TLabelType = LabelType;


        DataSetLoaderStats() = default;

        size_t examples = size_t();                          ///< Number of examples in data set
        
        IndexType featuresMinIndex = IndexType();            ///< Minimum feature index
        IndexType featuresMaxIndex = IndexType();            ///< Maximum feature index

        LabelType labelsMinValue = LabelType();              ///< Minimum label index
        LabelType labelsMaxValue = LabelType();              ///< Maximum label index

        //std::set<IndexType> features;         ///< Feature indices
        //std::set<LabelType> labels;           ///< Label Values

        bool ok = true;                                      ///< Status that all is ok and valid
    };

    template <class DesignMat,class ResponseVec, class PresentVec, class VecWeights>
    struct DataSet
    {
        typedef DesignMat   TDesignMat;
        typedef ResponseVec TResponseVec;
        typedef VecWeights  TVecWeights;

        TDesignMat train_samples_tr;           ///< Samples per column. Rows are predictor variables.
        TResponseVec train_outputs;            ///< Samples by rows. Labels or response or output of solved examples.
        bool has_intercept_term = false;       ///< Data set has intercept term in last (extra) Row.

        /** Total number of samples in data set
        * @return number of samples
        */
        size_t totalSamples() const {
            return train_samples_tr.columns();
        }

        /** Total number of attributes for tabular data including possibly intercept term
        * @return number of attributes
        */
        size_t numberOfAttributesForSamples() const {
            return train_samples_tr.rows();
        }

        /** Does design matrix in last column contains intercept term
        * @return status
        */
        bool hasInterceptTerm() const {
            return has_intercept_term;
        }

        /** Make data set completely empty
        */
        void makeEmpty() {
            train_samples_tr = TDesignMat();
            train_outputs = TResponseVec();
            has_intercept_term = false;
        }

        /** Split data set into several groups
        * @param samplesPerGroup is a vector which specify indicies of samples used by each group
        * @return array of data set
        */
        template <class IndexType>
        std::vector<DataSet> splitDataset(const std::vector<std::vector<IndexType>>& samplesPerGroup)
        {
            size_t groups     = samplesPerGroup.size();
            size_t attributes = numberOfAttributesForSamples();

            std::vector<DataSet> datasets(samplesPerGroup.size());

            // TODO 3: Logic can be parallelized across several threads
            for (size_t i = 0; i < groups; ++i)
            {
                auto& samples = samplesPerGroup[i];
                size_t samplesCount = samples.size();

                DataSet& curDataset = datasets[i];

                curDataset.train_samples_tr = TDesignMat(attributes, samplesCount);
                curDataset.train_outputs = TResponseVec(samplesCount);
                curDataset.has_intercept_term = has_intercept_term;

                // filling index
                IndexType fInd = IndexType();

                for (size_t j = 0; j < samplesCount; ++j, ++fInd)
                {
                    IndexType sample_index_in_parent = samples[j];

                    size_t parentDesigMatrixTrOffset = train_samples_tr.template getFlattenIndexFromColumn</*i*/0>(sample_index_in_parent);
                    
                    size_t childDesigMatrixTrOffset  = curDataset.train_samples_tr.template getFlattenIndexFromColumn</*i*/0>(fInd);

                    dopt::CopyHelpers::copy(&curDataset.train_samples_tr.matrixByCols[childDesigMatrixTrOffset],
                                            &train_samples_tr.matrixByCols[parentDesigMatrixTrOffset], attributes);

                    curDataset.train_outputs[fInd] = train_outputs[sample_index_in_parent];
                }
            }
            
            return datasets;
        }

        /** Concatenate datasets from all clients into a singl big dataset
        * @param datasets_per_clients vector of datasets from all clients
        * @return Results datasets which contains number of samples equal to sum of samples from all datasets, and number of attributes equal to max number of attributes from all datasets.
        */
        static DataSet concatDatasets(const std::vector<DataSet>& datasets_per_clients)
        {
            if (datasets_per_clients.empty())
                return DataSet();

            size_t totalSamples = 0;
            size_t totalAttributes = 0;

            for (auto i = datasets_per_clients.begin(); i != datasets_per_clients.end(); ++i)
            {
                totalSamples += i->totalSamples();
                size_t attrs = i->numberOfAttributesForSamples();
                if (attrs > totalAttributes)
                    totalAttributes = attrs;
            }

            DataSet result;
            
            // result.weights = DataSet::TVecWeights(totalSamples);

            result.train_samples_tr = DataSet::TDesignMat(totalAttributes, totalSamples);
            result.train_outputs    = DataSet::TResponseVec(totalSamples);
            result.has_intercept_term = datasets_per_clients[0].has_intercept_term;

            // filling index
            size_t fInd = size_t();
            
            // Scan through all client datasets
            for (auto i = datasets_per_clients.begin(); i != datasets_per_clients.end(); ++i)
            {
                const DataSet& curDataset = *i;

                // All dataset should have or not have intercept term
                assert(result.has_intercept_term == curDataset.has_intercept_term);

                size_t samplesCurrent = curDataset.totalSamples();

                // Scan sample by sample for each client
                for (size_t j = 0; j < samplesCurrent; ++j, ++fInd)
                {
                    // result.weights.set(fInd, curDataset.weights.get(j));
                    
                    size_t readOffset  = curDataset.train_samples_tr.template getFlattenIndexFromColumn</*i*/0> (j);
                    
                    size_t writeOffset = result.train_samples_tr.template getFlattenIndexFromColumn</*i*/0> (fInd);

                    // Copy a single samples features (they stoted in columns inside train_samples_tr matrix)
                    dopt::CopyHelpers::copy(
                                            &result.train_samples_tr.matrixByCols[writeOffset],
                                            &curDataset.train_samples_tr.matrixByCols[readOffset],
                                            curDataset.train_samples_tr.rows() 
                                           );
                    // Semantically:  result.train_samples_tr.setColumn(fInd, curDataset.train_samples_tr.getColumn(j));

                    // Copy Label
                    result.train_outputs[fInd] = curDataset.train_outputs[j];
                }
            }

            return result;
        }

        /** Fill data set randomly.
        * @remark Useful for checking splitting of the data set
        */
        template<class DatasetType>
        bool fillDatasetRandomly(DatasetType& trainset, 
                                 bool setWeights = true, 
                                 bool setDesignMatrix = true, 
                                 bool setLabels = true)
        {
            dopt::RandomGenRealLinear rnd(1.0, 10.0);

            // if (setWeights)
            //    trainset.weights.setAllRandomly(rnd);
            if (setDesignMatrix)
                trainset.train_samples.matrixByCols.setAllRandomly(rnd);
            if (setLabels)
                trainset.train_outputs.setAllRandomly(rnd);

            return true;
        }

        /** Check split consistency.
        * @param datasets_per_clients for each client
        * @param examplesPerClient train samples per client which stores indicies into original data set
        * @param trainset original data set which has been split
        * @return true if split is consistent
        * @remark First  - number of samples for each client(i) is exactly equal to examplesPerClient[i]
        * @remark Second - the copied data set records for each client is exactly the same as records in original data set.
        * @remark Third - checking that in union all indicies from clients and skipped indicies in fact are indicies for data set.
        */
        template<class IndexType, class DatasetType>
        bool checkSplitConsistency(const std::vector<DatasetType>& datasets_per_clients,
                                   const std::vector<std::vector<IndexType>>& examplesPerClient,
                                   const std::vector<IndexType>& skippedExamples,
                                   const DatasetType& trainset)
        {
            bool res = true;
            std::unordered_set<IndexType> allIndicies;
            // Put all indicies from all clients into allIndicies.
            //   Sanity Check 1: Clients do not have repeating indicies
            {
                size_t groupNum = 0;

                for (auto i = examplesPerClient.begin(); i != examplesPerClient.end(); ++i, ++groupNum)
                {
                    size_t curSize = allIndicies.size();

                    for (auto j = i->begin(); j != i->end(); ++j)
                    {
                        allIndicies.insert(*j);
                    }

                    size_t newSize = allIndicies.size();

                    if (newSize - curSize != i->size()) {
                        std::cerr << "Error in splitting. Some indicies are repeating for group #" << groupNum << '\n';
                        res &= false;
                    }
                }
            }
            //   Sanity Check 2: After reshuffling dataset indicies are sequentially copied into each group correctly.
            {
                size_t groupNum = 0;
                for (auto i = examplesPerClient.begin(); i != examplesPerClient.end(); ++i, ++groupNum)
                {
                    size_t localInd = 0;
                    for (auto j = i->begin(); j != i->end(); ++j, ++localInd)
                    {
                        size_t globalIndex = *j;

                        //double weight_diff = trainset.weights[globalIndex] - datasets_per_clients[groupNum].weights[localInd];
                        double train_outputs_diff = trainset.train_outputs[globalIndex] - datasets_per_clients[groupNum].train_outputs[localInd];
                        double train_samples_diff = (trainset.train_samples_tr.getColumn(globalIndex) - datasets_per_clients[groupNum].train_samples_tr.getColumn(localInd)).vectorL2Norm();

                        if (
                            //fabs(weight_diff) > 0.0 || 
                            fabs(train_outputs_diff) > 0.0 || 
                            fabs(train_samples_diff) > 0.0
                            )
                        {
                            std::cerr << "Error in splitting. Something has not been copied in a correct way for group #" << groupNum << '\n';
                            res &= false;
                        }
                    }
                }
            }

            // Add skiped indicies
            for (auto i = skippedExamples.begin(); i != skippedExamples.end(); ++i)
            {
                allIndicies.insert(*i);
            }

            // Sanity Check - 3: If add skipp indicies to previously collected indicies cardinality of such set is equal to total numer of datapoints.

            size_t trainDatasetSize = trainset.totalSamples();

            // Sanity Check - 4: All indicies are presented in "allIndicies" set.
            for (size_t i = 0; i < trainDatasetSize; ++i)
            {
                if (allIndicies.find(i) == allIndicies.end())
                {
                    std::cerr << "Error in splitting. The record " << i << " has been lost\n";
                    res &= false;
                }
            }

            return res;
        }

        /** Debug print of train dataset into out stream.
        * @param out output stream
        * @param lineSep separator between lines
        * @param fieldSep separator between fields
        * @param attrValSep separator between attribute/value pair
        */
        template<class text_out_steam>
        void printInTextFormat(text_out_steam& out, const char* lineSep = "\n", const char* fieldSep = " ", const char* attrValSep = ":") const
        {
            size_t samples = totalSamples();
            size_t attrs = numberOfAttributesForSamples();

            for (size_t i = 0; i < samples; ++i)
            {
                if (train_outputs[i] > 0)
                    out << "+";
                
                out << train_outputs[i];
                
                out << fieldSep;

                for (size_t j = 0; j < attrs; ++j)
                {
                    if (train_samples_tr.get(j, i) != 0)
                    {
                        out << j << attrValSep << train_samples_tr.get(j, i) << fieldSep;
                    }
                }
                out << lineSep;
            }
        }
    };
    
    /** Dataset loader from text file
    */
    template<class FLineSeparator, class FFieldSeparator, 
             class FAttributeValueSeparator, class FSkipSymbols>
    class TextFileDataSetLoader
    {
    private:
        Data& dataInput;                                  ///< Input source

        FLineSeparator  lineSeparator;                    ///< Check Line separator
        FFieldSeparator fieldSeparatorInLine;             ///< Check field separator
        FAttributeValueSeparator attributeValueSeparator; ///< Feature Index and Value separator
        FSkipSymbols             symbolSkipping;          ///< Skip symbols

    public:
        TextFileDataSetLoader(Data& theDataInput,
                              FLineSeparator theLineSeparator,
                              FFieldSeparator theFieldSeparatorInLine,
                              FAttributeValueSeparator theAttributeValueSeparator,
                              FSkipSymbols theSymbolSkipping)
        : lineSeparator(theLineSeparator)
        , fieldSeparatorInLine(theFieldSeparatorInLine)
        , attributeValueSeparator(theAttributeValueSeparator)
        , symbolSkipping(theSymbolSkipping)
        , dataInput(theDataInput)
        {}

        ~TextFileDataSetLoader() = default;

        /** Count statistics of data set for allocate memory to store it and process it
        */
        template <bool restartDataSource, bool omitSymbolSkipCheck, class IndexType, class LabelType>
        DataSetLoaderStats<IndexType, LabelType> computeStatisticsWithOneDataPass()
        {
            DataSetLoaderStats<IndexType, LabelType> res = {};

            if constexpr (restartDataSource)
            {
                dataInput.rewindToStart();
            }
            
            if (dataInput.getResidualLength() == 0)                
            {
                res.ok = false;
                return res;
            }

            constexpr size_t kExpectedLabels = 2;
            size_t uniqueLabels = 0;
            size_t lines = 0;

            // Draft memory used during parsing
            dopt::MutableData lineData;
            std::vector<std::string_view> feature_value;
            feature_value.reserve(2);

            std::vector<std::string_view> fields;

            // Optimize copying
            size_t backlogSize = 0;
            uint8_t* backlogRawPtr = nullptr;

            // Flags for actions
            bool process_line = false;
            bool exit_the_loop = false;

            for (; !exit_the_loop; )
            {  
                char symbol = dataInput.getCharacter();
                
                if (!dataInput.isLastGetSuccess())
                {
                    if (backlogSize > 0)
                    {
                        // Save raw pointer. Minus backlog.
                        backlogRawPtr = dataInput.getPtrToResidual() - backlogSize;
                        process_line = true;
                    }
                    else if (!lineData.isEmpty())
                    {
                        process_line = true;
                    }
                    // process_line = false;
                    exit_the_loop = true;
                }
                else if (!omitSymbolSkipCheck && symbolSkipping(symbol))
                {
                    if (backlogSize > 0)
                    {
                        // Flush backlog completely
                        // Minus backlog and minus one recently received symbol
                        backlogRawPtr = dataInput.getPtrToResidual() - backlogSize - 1;
                        lineData.putBytes(backlogRawPtr, backlogSize);

                        backlogSize = 0;
                        backlogRawPtr = nullptr;
                    }
                    
                    // Nothing todo with with lineData
                    continue;
                }
                else if (lineSeparator(symbol))
                {
                    if (backlogSize > 0)
                    {
                        // Save raw pointer. Minus backlog and minus one recently received symbol
                        backlogRawPtr = dataInput.getPtrToResidual() - backlogSize - 1;
                        process_line = true;
                    }
                    else if (!lineData.isEmpty())
                    {
                        process_line = true;
                    }                   
                }
                else [[likely]]
                {
                    backlogSize++;
                }
                    
                // logic for process the current input line
                if (process_line)
                {                    
                    const char* lineDataPtr = nullptr;
                    size_t lineDataSize = 0;

                    if (!lineData.isEmpty())
                    {
                        // Flush backlog completely
                        lineData.putBytes(backlogRawPtr, backlogSize);

                        // Setup raw pointers
                        lineDataPtr = (char*)lineData.getPtr();
                        lineDataSize = lineData.getFilledSize();
                    }
                    else
                    {
                        // Setup raw pointers. Memory Copy Optimization
                        lineDataPtr = (char*)backlogRawPtr;
                        lineDataSize = backlogSize;
                    }                        

                    // Ignore empty lines
                    if (lineDataSize > 0)
                    {
                        // Reset backlog
                        backlogSize = 0;
                        backlogRawPtr = nullptr;

                        // Create string view for current line
                        std::string_view lineStr(lineDataPtr, lineDataSize);

                        fields.clear();
                        dopt::string_utils::splitToSubstrings(fields, 
                                                              lineStr, 
                                                              fieldSeparatorInLine);

                        size_t fieldsLen = fields.size();

                        if (fieldsLen > 0)
                        {
                            for (size_t i = 0; i < fieldsLen; ++i)
                            {
                                feature_value.clear();

                                dopt::string_utils::splitToSubstrings(feature_value, 
                                                                      /*keyvalue*/fields[i], 
                                                                      attributeValueSeparator);

                                if (feature_value.size() == 1)
                                {
                                    LabelType labelValue = LabelType();
                                    if (!dopt::string_utils::fromString(labelValue, feature_value[0])) {
                                        std::cerr << "Error during conversion of lable: " << feature_value[0] << ". Parsing Line: " << res.examples + 1 << '\n';
                                    } else {
                                        //res.labels.insert(labelValue);

                                        // Update Label Info
                                        if (res.examples == 0)
                                        {
                                            res.labelsMinValue = labelValue;
                                            res.labelsMaxValue = labelValue;
                                            uniqueLabels = 1;
                                        }
                                        else
                                        {
                                            if (labelValue != res.labelsMinValue && labelValue != res.labelsMaxValue)
                                            {
                                                uniqueLabels++;

                                                if (labelValue < res.labelsMinValue)
                                                    res.labelsMinValue = labelValue;
                                                else if (labelValue > res.labelsMaxValue)
                                                    res.labelsMaxValue = labelValue;
                                            }
                                        }
                                    }
                                }
                                else if (feature_value.size() == 2)
                                {
                                    IndexType index;
                                    if (dopt::string_utils::fromString(index, feature_value[0]))
                                    {
                                        // res.features.insert(index);

                                        // Update Features Info
                                        if (res.examples == 0)
                                        {
                                            res.featuresMinIndex = index;
                                            res.featuresMaxIndex = index;
                                        }
                                        else
                                        {
                                            if (index < res.featuresMinIndex)
                                                res.featuresMinIndex = index;
                                            else if (index > res.featuresMaxIndex)
                                                res.featuresMaxIndex = index;
                                        }
                                    }
                                    else
                                    {
                                        res.ok = false;
                                        exit_the_loop = true;
                                        std::cerr << "Error during parsing. Line: " << res.examples + 1 << ". Can not parse " << feature_value[0] << " for purpose of converting to unsigned integer.\n";
                                    }
                                }
                                else
                                {
                                    // skip (error)
                                    std::cerr << "Warning during parsing. Line: " << res.examples + 1 << ". Can not parse " << fields[i] << " for purpose of obtaing feature_index:value of label.\n";
                                }
                            }
                            
                            res.examples += 1;
                            // fields.clear();
                        }

                        // End of process line
                        // Rewind to start position write head
                        lineData.rewindToStart();
                    }
                    
                    // Reset the flag
                    process_line = false;
                }
            }
                                            
            //if (res.labels.size() != 2)
            if (uniqueLabels != kExpectedLabels)
            {
                std::cout << "Specified dataset labels for " << uniqueLabels << " classes. And you need to have a dataset with " << kExpectedLabels << " classes.\n";
                res.ok = false;
                return res;
            }

            return res;
        }

        template <bool restartDataSource, bool omitSymbolSkipCheck, class DataSet, class TextDataSetLoaderStats>
        size_t createDataset(DataSet& dataset, const TextDataSetLoaderStats& stats, /*CreateDatasetFlags*/ int flags)
        {
            auto min_index = stats.featuresMinIndex;
            auto max_index = stats.featuresMaxIndex;
            
            //auto max_index = *(stats.features.rbegin());
#if 0
            auto max_index = *(stats.features.begin());
            for (auto i = stats.features.begin(); i != stats.features.end(); ++i)
            {
                if (*i > max_index)
                    max_index = *i;
            }
#endif
            // assert(stats.labels.size() > 0);

            //typename TextDataSetLoaderStats::TLabelType minValueLabel = *(stats.labels.begin());
            //typename TextDataSetLoaderStats::TLabelType maxValueLabel = *(stats.labels.rbegin());

            typename TextDataSetLoaderStats::TLabelType minValueLabel = stats.labelsMinValue;
            typename TextDataSetLoaderStats::TLabelType maxValueLabel = stats.labelsMaxValue;

            typedef typename DataSet::TResponseVec::TElementType TActualLabelValue;

            constexpr bool kLabelsIsFloatNumber = std::is_same<TActualLabelValue, float >::value ||
                                                  std::is_same<TActualLabelValue, double>::value;
            
            TActualLabelValue avgValueLabel = (TActualLabelValue(minValueLabel) + TActualLabelValue(maxValueLabel)) / TActualLabelValue(2);

            auto columns_in_design_matrix = max_index - min_index + 1;

            if (flags & CreateDatasetFlags::eAddInterceptTerm)
            {
                columns_in_design_matrix += 1;
                dataset.has_intercept_term = true;
            }
            else
            {
                dataset.has_intercept_term = false;
            }

            dataset.train_samples_tr   = typename DataSet::TDesignMat(columns_in_design_matrix, stats.examples);
            dataset.train_outputs      = typename DataSet::TResponseVec(stats.examples);

            typename DataSet::TDesignMat::TElementType* train_samples_tr_raw = dataset.train_samples_tr.matrixByCols.data();
            size_t examples_column_offset = 0;
            size_t LDA = dataset.train_samples_tr.LDA;

            typename DataSet::TResponseVec::TElementType* train_outputs_raw = dataset.train_outputs.data();

            if constexpr (restartDataSource)
            {
                dataInput.rewindToStart();
            }

            if (dataInput.isEmpty())
            {
                return 0;
            }

            size_t examples = 0;

            // Draft memory used during parsing
            dopt::MutableData lineData;
            std::vector<std::string_view> feature_value_pair;
            feature_value_pair.reserve(2);
            std::vector<std::string_view> fields;
            //fields.reserve( dopt::minimum(64, (stats.featuresMaxIndex - stats.featuresMinIndex)) );
            
            // Optimize copying
            size_t backlogSize = 0;
            uint8_t* backlogRawPtr = nullptr;
            
            // Flags for actions
            bool process_line = false;
            bool exit_the_loop = false;

            for (; !exit_the_loop; )
            {                
                char symbol = dataInput.getCharacter();

                if (!dataInput.isLastGetSuccess())
                {
                    if (backlogSize > 0)
                    {
                        // Save raw pointer. Minus backlog.
                        backlogRawPtr = dataInput.getPtrToResidual() - backlogSize;
                        process_line = true;
                    }
                    else if (!lineData.isEmpty())
                    {
                        process_line = true;
                    }
                    // process_line = false;
                    exit_the_loop = true;
                }
                else if (!omitSymbolSkipCheck && symbolSkipping(symbol))
                {
                    if (backlogSize > 0)
                    {
                        // Flush backlog completely
                        // Minus backlog and minus one recently received symbol
                        backlogRawPtr = dataInput.getPtrToResidual() - backlogSize - 1;
                        lineData.putBytes(backlogRawPtr, backlogSize);

                        backlogSize = 0;
                        backlogRawPtr = nullptr;
                    }
                    
                    // Nothing todo with with lineData
                    continue;
                }
                else if (lineSeparator(symbol))
                {
                    if (backlogSize > 0)
                    {
                        // Save raw pointer. Minus backlog and minus one recently received symbol
                        backlogRawPtr = dataInput.getPtrToResidual() - backlogSize - 1;
                        process_line = true;
                    }
                    else if (!lineData.isEmpty())
                    {
                        process_line = true;
                    }
                }
                else [[likely]]
                {
                    backlogSize++;
                }
                
                // logic for process the current input line
                if (process_line)
                {
                    const char* lineDataPtr = nullptr;
                    size_t lineDataSize = 0;

                    if (!lineData.isEmpty())
                    {
                        // Flush backlog completely (if there is a backlog)
                        lineData.putBytes(backlogRawPtr, backlogSize);

                        // Setup raw pointers
                        lineDataPtr = (char*)lineData.getPtr();
                        lineDataSize = lineData.getFilledSize();
                    }
                    else
                    {
                        // Setup raw pointers. Memory Copy Optimization.
                        lineDataPtr = (char*)backlogRawPtr;
                        lineDataSize = backlogSize;
                    }

                    // Ignore empty lines
                    if (lineDataSize > 0)
                    {
                        // Reset backlog
                        backlogSize = 0;
                        backlogRawPtr = nullptr;
                    
                        std::string_view lineStr(lineDataPtr, lineDataSize);
                    
                        fields.clear();
                        dopt::string_utils::splitToSubstrings(fields, lineStr, fieldSeparatorInLine);

                        size_t fieldsLen = fields.size();

                        if (fieldsLen > 0)
                        {
                            for (size_t i = 0; i < fieldsLen; ++i)
                            {
                                // std::string keyvalue = { fields[i].begin(), fields[i].end() };
                                feature_value_pair.clear();
                                dopt::string_utils::splitToSubstrings(feature_value_pair, /*keyvalue*/fields[i], attributeValueSeparator);

                                if (feature_value_pair.size() == 1)
                                {
                                    // label
                                    bool parse_label = dopt::string_utils::fromString(train_outputs_raw[examples], feature_value_pair[0]);

                                    if (!parse_label)
                                    {
                                        std::cerr << "Error during parsing. Line: " << examples + 1 << ". Can not parse "
                                            << feature_value_pair[0] << " for purpose of converting to response variable.\n";
                                    }

                                    // remap label if needed
                                    if (flags & CreateDatasetFlags::eMakeRemappingForBinaryLogisticLoss)
                                    {
                                        if constexpr (kLabelsIsFloatNumber)
                                        {
                                            if (train_outputs_raw[examples] < avgValueLabel)
                                            {
                                                train_outputs_raw[examples] = TActualLabelValue(-1);
                                            }
                                            else
                                            {
                                                train_outputs_raw[examples] = TActualLabelValue(+1);
                                            }
                                        }
                                        else
                                        {
                                            if (dopt::abs(train_outputs_raw[examples] - TActualLabelValue(minValueLabel)) <
                                                dopt::abs(train_outputs_raw[examples] - TActualLabelValue(maxValueLabel))
                                                )
                                            {
                                                train_outputs_raw[examples] = TActualLabelValue(-1);
                                            }
                                            else
                                            {
                                                train_outputs_raw[examples] = TActualLabelValue(+1);
                                            }
                                        }
                                    }
                                }
                                else if (feature_value_pair.size() == 2)
                                {
                                    // (feature index, value) pair
                                    typename TextDataSetLoaderStats::TIndexType feature_index;
                                    typename DataSet::TDesignMat::TElementType feature_value;

                                    bool parse_feature_index = dopt::string_utils::fromString(feature_index, feature_value_pair[0]);
                                    feature_index -= min_index;

                                    if (!parse_feature_index)
                                    {
                                        std::cerr << "Error during parsing. Line: " << examples + 1 << ". Can not parse: "
                                            << feature_value_pair[0] << " for purpose of converting to feature index.\n";
                                    }

                                    bool parse_feature_value = dopt::string_utils::fromString(feature_value, feature_value_pair[1]);
                                    if (!parse_feature_value)
                                    {
                                        std::cerr << "Error during parsing. Line: " << examples + 1 << ". Can not parse: "
                                            << feature_value_pair[1] << " for purpose of converting to feature value.\n";
                                    }

                                    // dataset.train_samples_tr.set(feature_index, examples, feature_value);
                                    *(train_samples_tr_raw + 
                                      examples_column_offset/*col-like*/ + 
                                      feature_index/*row*/) = feature_value;
                                }
                                else
                                {
                                    std::cerr << "Error during parsing. Line: " << examples + 1 << ". More then one feature_index:value pair has been found.\n";
                                }                            
                            }

                            // Optional setup of intercept term
                            if (flags & CreateDatasetFlags::eAddInterceptTerm)
                            {
                                // dataset.train_samples_tr.set(columns_in_design_matrix - 1, examples, 1);
                                *(train_samples_tr_raw +
                                    examples_column_offset       /*col-like*/ +
                                    columns_in_design_matrix - 1 /*row*/) = 1;
                            }

                            examples += 1;
                            examples_column_offset += LDA;
                        }

                        // End of process_line
                        // Rewind to start position write head
                        lineData.rewindToStart();
                    }
                
                    // Reset the flag
                    process_line = false;
                }
            }

            // dataset.weights.setAll(1);
            //dataset.train_samples_tr = dataset.train_samples.getTranspose();

            assert(examples == stats.examples);

            return examples;
        }
    };
}
