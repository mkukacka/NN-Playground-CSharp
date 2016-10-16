using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_Basic_Tests
{
    class TestConfiguration
    {
        public List<Tuple<double[], double[]>> TrainingSet;

        public int InputSize;
        public int OutputSize;
        public int HiddenLayerSize;
        public double ActivationParameter;

        /*
         * var trainingSet = new List<Tuple<double[], double[]>>{
                new Tuple<double[], double[]>(new double[]{0.0, 0.5, 1.0 }, new double[]{1.0, 0.0, 0.0}),
                new Tuple<double[], double[]>(new double[]{1.0, 0.5, 0.0 }, new double[]{0.0, 1.0, 0.0}),
                new Tuple<double[], double[]>(new double[]{0.0, 0.1, 0.0 }, new double[]{0.0, 0.0, 1.0})
            };

            var inputSize = 3;
            var outputSize = 3;
            var hiddenLayerSize = 4;
            var activationParameter = 0.5;
         */
    }
}
