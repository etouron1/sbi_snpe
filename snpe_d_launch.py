import sbibm 
import argparse
from sbibm.algorithms import snpe


#all tasks
print(sbibm.get_available_tasks())

def sample_posterior(args):
    """generate separated csv files with the posterior samples for each of the algos"""

    task = sbibm.get_task(args.model)
    snpe(task=task, neural_net=args.neural_net_snpe, variant="D", num_samples=args.n_posterior, num_observation=args.nb_obs, num_simulations=args.n_train, num_rounds=args.n_sequential, training_batch_size=args.batch_size_training)



def main(args):
    sample_posterior(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Simulator experience'
    )
    parser.add_argument('--model', '-m', type=str, default="gaussian_linear",
                        help="The model to use among : 'gaussian_linear', 'bernoulli_glm', 'slcp', 'two_moons', 'gaussian_linear_uniform', 'gaussian_mixture', 'slcp_distractors', 'bernoulli_glm_raw'")
    parser.add_argument('--snre_type', '-snre_t', type=str, default="B",
                        help='The type of SNRE A or B')
    parser.add_argument('--n_train', '-ntr', type=int, default=5000,
                        help='Number of train samples to make')
    parser.add_argument('--n_sequential', '-ns', type=int, default=1,
                        help='Number of rounds to do in sequential algo')
    parser.add_argument('--nb_obs', '-nobs', type=int, default=1,
                        help='Number of observations x to have')
    parser.add_argument('--neural_net_snpe', '-nnsnpe', type=str, default="nsf",
                        help='The neural network for the posterior estimator among maf / mdn / made / nsf')
    parser.add_argument('--variant_snpe', '-vsnpe', type=str, default="C",
                        help="The variant of SNPE among 'A' or 'C'")
    parser.add_argument('--variant_snre', '-vsnre', type=str, default="B",
                        help="The variant of SNRE among 'A', 'B', 'C' or 'D' for BNRE")
    parser.add_argument('--neural_net_snle', '-nnsnle', type=str, default="nsf",
                        help='The neural network for the likelihood estimator among maf / mdn / made / nsf')
    parser.add_argument('--neural_net_snre', '-nnsnre', type=str, default="resnet",
                        help='The neural network for the ratio estimator among linear / mlp / resnet')
    parser.add_argument('--batch_size_simulator', '-bss', type=int, default=1000,
                        help='Bacth size for the simulatior')
    parser.add_argument('--batch_size_training', '-bstr', type=int, default=10000,
                        help='Bacth size for the simulator')
    parser.add_argument('--n_posterior', '-np', type=int, default=10000,
                        help='Number of posterior samples theta')

    args = parser.parse_args()
    main(args)
    
