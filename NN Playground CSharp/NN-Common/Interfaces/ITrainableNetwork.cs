using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_Common.Interfaces
{
    public interface ITrainableNetwork
    {
        void Train(double[] input, double[] targetOutput, double learningParam);
        double GetNetworkTotalError(double[] input, double[] targetOutput);

    }
}
