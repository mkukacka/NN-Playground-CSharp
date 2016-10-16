using NN_Basic;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NN_Common.Interfaces;

namespace NN_Basic_Tests
{
    [TestFixture]
    public class Simple2layerNetworkTests
    {

        [Test]
        public void SimpleNetworkProcessingTest()
        {
            // just to see whether everything is going ok - basic implementation check

            var config = this.CreateSimpleTestConfiguration();

            var network = new Simple2layerNetwork(config.InputSize, config.HiddenLayerSize, config.OutputSize, config.ActivationParameter);

            var output = network.GetNetworkOutput(config.TrainingSet[0].Item1);
        }

        [Test]
        public void SimpleLearningTest()
        {
            var config = this.CreateSimpleTestConfiguration();
            
            var network = new Simple2layerNetwork(config.InputSize, config.HiddenLayerSize, config.OutputSize, config.ActivationParameter);

            var learningParam = 1.0;
            int numLearningCycles = 50;

            RunNetworkTraining(network, config, learningParam, numLearningCycles);
        }

        [Test]
        public void TestLearningWithMoment()
        {
            var config = this.CreateSimpleTestConfiguration();

            double learningMoment = 0.8;

            var network = new Simple2layerNetwork_withMoment(config.InputSize, config.HiddenLayerSize, config.OutputSize,
                config.ActivationParameter, learningMoment: learningMoment);

            var learningParam = 1.0;
            int numLearningCycles = 50;

            RunNetworkTraining(network, config, learningParam, numLearningCycles);
        }

        [Test]
        public void TestXORLearningWithMoment()
        {
            var config = this.CreateConfigForLearningXOR();

            config.ActivationParameter = 1.0;
            double learningMoment = 0.8;

            var network = new Simple2layerNetwork_withMoment(config.InputSize, config.HiddenLayerSize, config.OutputSize,
                config.ActivationParameter, learningMoment: learningMoment);

            var learningParam = 1.0;
            int numLearningCycles = 500;

            RunNetworkTraining(network, config, learningParam, numLearningCycles);
        }

        void RunNetworkTraining(ITrainableNetwork network, TestConfiguration config, double learningParam, double numEpochs)
        {
            double err;
            for (int i = 0; i < numEpochs; i++)
            {
                // measure the error before the learning, on the whole training set
                err = 0.0;

                // TODO randomize the order of training samples 
                foreach (var trainSample in config.TrainingSet)
                {
                    err += network.GetNetworkTotalError(trainSample.Item1, trainSample.Item2);
                }
                Console.WriteLine("Step {0}, total network error: {1}", i, err);

                // train all samples
                foreach (var trainSample in config.TrainingSet)
                {
                    network.Train(trainSample.Item1, trainSample.Item2, learningParam);
                }
            }

            // measure final error of the network
            err = 0.0;
            foreach (var trainSample in config.TrainingSet)
            {
                err += network.GetNetworkTotalError(trainSample.Item1, trainSample.Item2);
            }
            Console.WriteLine("Final total network error: {0}", err);
        }

        #region Network setup factories

        TestConfiguration CreateSimpleTestConfiguration()
        {
            var config = new TestConfiguration();

            config.TrainingSet = new List<Tuple<double[], double[]>>{
                new Tuple<double[], double[]>(new double[]{0.0, 0.5, 1.0 }, new double[]{1.0, 0.0, 0.0}),
                new Tuple<double[], double[]>(new double[]{1.0, 0.5, 0.0 }, new double[]{0.0, 1.0, 0.0}),
                new Tuple<double[], double[]>(new double[]{0.0, -1.0, 0.0 }, new double[]{0.0, 0.0, 1.0})
            };

            config.InputSize = 3;
            config.OutputSize = 3;
            config.HiddenLayerSize = 4;
            config.ActivationParameter = 0.5;

            return config;
        }

        TestConfiguration CreateConfigForLearningXOR()
        {
            var config = new TestConfiguration();

            config.TrainingSet = new List<Tuple<double[], double[]>>{
                new Tuple<double[], double[]>(new double[]{0.0, 0.0}, new double[]{0.0}),
                new Tuple<double[], double[]>(new double[]{1.0, 0.0}, new double[]{1.0}),
                new Tuple<double[], double[]>(new double[]{0.0, 1.0}, new double[]{1.0}),
                new Tuple<double[], double[]>(new double[]{1.0, 1.0}, new double[]{0.0})
            };

            config.InputSize = 2;
            config.OutputSize = 1;
            config.HiddenLayerSize = 3;
            config.ActivationParameter = 1.0;

            return config;
        }

        #endregion
    }
}
