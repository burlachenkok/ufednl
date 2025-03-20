#include "dopt/linalg_vectors/include/VectorND_Raw.h"
#include "dopt/linalg_matrices/include/MatrixNMD.h"
#include "dopt/random/include/RandomGenMersenne.h"
#include "dopt/random/include/Shuffle.h"

#include "dopt/cmdline/include/CmdLineParser.h"

#include "dopt/fs/include/StringUtils.h"
#include "dopt/fs/include/FileSystemHelpers.h"

#include "dopt/copylocal/include/MutableData.h"

#include "dopt/timers/include/HighPrecisionTimer.h"

#include <iostream>

// Cmdline example:
//  --dimension 10 --samples 10 --seed 123 --feature-generator u_in_0_1 --classes "-1,1" --classes_fractions "0.5,0.5" --out my.txt --info 

int main(int argc, char** argv)
{
    dopt::HighPrecisionTimer timer;

    dopt::CmdLine cmdline(argc, argv);

    if (argc == 1)
    {
        std::cout << "Generating dataset for binary classification" << '\n';
        std::cout << "Example of command line:" << '\n';
        std::cout << "  --dimension 2 --samples 10 --seed 123 --feature - generator u_in_0_1 --classes \"-1,1\" --classes_fractions \"0.5,0.5\" --out my.txt --info" << '\n';
    }

    //=======================================================================================
    // PARSE COMMAND LINE GENERAL FLAGS
    //=======================================================================================
    int d = 0;
    
    if (!cmdline.getIntArgByName(d, "dimension")) {
        std::cout << "Please specify dimension of the problem with as '--dimension 2' " << '\n';
        return -1;
    }
    
    int samples = 0;
    if (!cmdline.getIntArgByName(samples, "samples")) {
        std::cout << "Please specify number of samples for the the problem with '--samples 10' " << '\n';
        return -1;
    }
    
    int seed = 123;
    if (!cmdline.getIntArgByName(seed, "seed")) {
        std::cout << "Please specify seed for data generation '--seed 123' " << '\n';
        return -1;
    }

    std::string fname = "";
    if (!cmdline.getStringArgByName(fname, "out")) {
        std::cout << "Please specify name of output file (or use \"-\" for stdout) as a resuly for data generation '--out my.txt' " << '\n';
        return -1;
    }
    
    const char* feature_generation_rules[] = { "u_in_0_1", "u_in_[-1_1]", "u_in_[-10_10]" };

    std::string_view feature_generation_rule = "";
    
    if (!cmdline.getStringViewArgByName(feature_generation_rule, "feature-generator")) {
        std::cout << "Please specify seed for data generation '--feature-generator' " << '\n';
        std::cout << "Supported rules:\n";
        for (const auto& r : feature_generation_rules) {
            std::cout << "  " << r << '\n';
        }
        
        return -1;
    }
    //======================================================================================
    // PARSE COMMAND LINE INFORMATION ABOUT CLASSES AND NUMBER OF SAMPLES PER EACH CLASS
    //=======================================================================================

    std::vector<int> classesVecInt;             // Class labels.   Typically "-1,1"
    std::vector<double> classesFractions;       // Class fraction. Typically "0.5,0.5"
    std::vector<int> classesFractionsInSamples; // Number of samples for each class
    
    {
        std::string_view classes = "";
        if (!cmdline.getStringViewArgByName(classes, "classes")) {
            std::cout << "Please specify classes labels --classes \"-1,1\" " << '\n';
            return -1;
        }
        
        std::vector<std::string_view> classes_vec = dopt::string_utils::splitToSubstrings(classes, ',');

        for (const auto& c : classes_vec) {
            int classInt = 0;
            if (!dopt::string_utils::fromString(classInt, c)) {
                std::cout << "Please specify classes labels in forms of integers" << '\n';
                return -1;
            }
            classesVecInt.push_back(classInt);
        }
    }
    
    {
        std::string_view classes_fractions = "";
        if (!cmdline.getStringViewArgByName(classes_fractions, "classes_fractions")) {
            std::cout << "Please specify classes fractions --classes_fractions \"0.25,0.75\" " << '\n';
            return -1;
        }
        
        std::vector<std::string_view> classes_fractions_vec = dopt::string_utils::splitToSubstrings(classes_fractions, ',');
        
        double cf_sum = 0.0;
        
        for (const auto& cf : classes_fractions_vec) 
        {
            double classFraction = 0;
            if (!dopt::string_utils::fromString(classFraction, cf)) 
            {
                std::cout << "Please specify classes fraction as double value" << '\n';
                return -1;
            }
            classesFractions.push_back(classFraction);
            cf_sum += classFraction;
        }
        
        int created_samples = 0;
        classesFractionsInSamples.resize(classesFractions.size());
        
        for (size_t i = 0; i < classesFractions.size(); ++i)
        {
            classesFractions[i] /= cf_sum;
            int cur_samples = 0;
            
            if (i == classesFractions.size() - 1)
            {
                cur_samples = samples - created_samples;
            } 
            else 
            {
                cur_samples = int(classesFractions[i] * samples);
            }

            classesFractionsInSamples[i] = cur_samples;
            created_samples += cur_samples;
        }
    }
    
    // Set seed for random generator
    dopt::RandomGenMersenne generator;
    generator.setSeed(seed);

    //================================================================================
    dopt::VectorNDRaw_i classes = dopt::VectorNDRaw_i::getUninitializedVector(d);
    dopt::MatrixNMD<dopt::VectorNDRaw_d> features(samples, d);
    
    // Generate features and classes
    double a = 0.0, b = 1.0;
    
    if (feature_generation_rule == "u_in_0_1")
    {
        a = 0.0;
        b = 1.0;
    }
    else if (feature_generation_rule == "u_in_[-1_1]")
    {
        a = -1.0; 
        b = 1.0;
    }
    else if (feature_generation_rule == "u_in_[-10_10]")
    {
        a = -10.0;
        b = 10.0;
    }
    else
    {
        std::cout << "Unsupported feature generation rule: " << feature_generation_rule << '\n';
        return -1;
    }
    //================================================================================
    {
        {
            double* restrict_ext rawFeaturesData = features.matrixByCols.data();
            size_t sz = features.matrixByCols.size();
            for (size_t i = 0; i < sz; ++i)
                rawFeaturesData[i] = generator.generateReal(a, b);
        }
        
        {
            int generate_samples = 0;
            int32_t* restrict_ext rawClassData = classes.data();

            for (size_t i = 0; i < classesFractions.size(); ++i)
            {
                for (size_t j = 0; j < classesFractionsInSamples[i]; ++j)
                {
                    rawClassData[generate_samples] = classesVecInt[i];
                    ++generate_samples;
                }
            }
            
            assert(generate_samples == samples);
        }
        
        dopt::shuffle(classes, generator);
    }
    //================================================================================
    // OUTPUT DATA TO FILE
    dopt::MutableData out;

    for (size_t i = 0; i < samples; ++i)
    {
        out.putIntegerAsAString(classes[i]);
        out.putCharacter(' ');
        
        for (size_t j = 0; j < d; ++j)
        {
            out.putUnsignedIntegerAsAString(j);
            out.putCharacter(':');

            std::string fvalue = dopt::string_utils::toString(features.getRaw(i, j));
            out.putString(fvalue, dopt::MutableData::PutStringFlags::ePutNoTerminator);
            out.putCharacter(' ');
        }
        out.putCharacter('\n');
    }
    
    if (fname != "-")
    {
        bool results_are_saved = dopt::FileSystemHelpers::saveFile(fname, out.getPtr(), out.getFilledSize());
        std::cout << "Results has been saved to: " << fname << (results_are_saved ? " [OK]" : " [FAILED]") << '\n';
    }
    else
    {
        std::cout.write( (char*)out.getPtr(), out.getFilledSize() );
    }

    double deltaMs = timer.getTimeMs();

    if (cmdline.isFlagSetuped("info"))
    {
        std::cout << "INFO" << '\n';
        std::cout << "=========================================================================\n";        
        std::cout << "Working Directory: " << dopt::FileSystemHelpers::getCwd() << '\n';
        std::cout << "Time spent to execution is " << deltaMs << " milliseconds\n" << '\n';
        std::cout << '\n';

        std::cout << "dimension: "      << d << '\n';
        std::cout << "samples: "        << samples << '\n';
        std::cout << "seed: "           << seed << '\n';
        std::cout << "out file name: "  << fname << '\n';
        std::cout << "=========================================================================\n";
    }

    return 0;
}
