import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import operator
import argparse
import sys
import matplotlib.pyplot as plt
plt.style.use('data/plots_paper.mplstyle')
import pathlib


class algs:
    def __init__(self, theta=67, alpha=3, sigma=2)

        assert(theta <= 147 and theta >=0)
        # True theta
        self.theta = theta
        self.alpha = alpha
        self.sigma = sigma

        # Load data
        self.train_data_rating = np.load('preproc/genres/train_data_rating.npy')
        #For regret computation
        self.test_data_rating = np.load('preproc/genres/test_data_rating.npy')
        # Test data
        self.test_data_df = pd.read_pickle('genres/test_data_with_id.npy')
        # Pulling out relevant columns for quicker lookup
        self.test_data = np.concatenate([np.array(test_data.Rating.tolist())[:,None], \
                        np.array(test_data.Genre_Col.tolist())[:, None], \
                        np.array(test_data.Meta_User_Col.tolist())[:, None]], \
                        axis=1)

        # Initialize mu [# meta-users, # genres]
        mu_ = np.zeros((147, 18))
        mu_test = np.zeros((147, 18))
        for i in range(len(self.train_data_rating)):
            mu_[i,:] = self.train_data_rating[i].values()
            mu_test[i,:] = self.test_data_rating[i].values()

        theta_set = list(range(147))

        # Clear low counts
        self.mu, self.mu_test, self.theta_set = self._clear_low_counts(mu_, mu_test, theta_set)
        self.numArms = mu.shape[1]
        self.bestArm = np.argmax(mu[self.theta, :])

        # Re-define theta as updated index - can be removed based on how theta is picked
        self.theta = self._index_of(self.theta)

    def _index_of(self):
        return np.where(self.theta_set == self.theta)[0][0]

    def _clear_low_counts(self, mu, mu_test, theta_set):
        #These have <250 counts: As selected from the analysis code
        remove_metau = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18,\
        20, 29, 34, 52, 55, 71, 73, 88, 94, 103, 109, 115, 124, 130, 131, 134, \
        135, 136, 137, 144,145]

        mu = np.delete(mu, (remove_metau), axis = 0)
        zero_counts_remove = np.where(mu == 0)[0]
        mu = np.delete(mu, (zero_counts_remove), axis = 0)

        mu_test = np.delete(mu_test, (remove_metau), axis = 0)
        mu_test = np.delete(mu_test, (zero_counts_remove), axis = 0)

        theta_set = np.delete(theta_set, remove_metau)
        theta_set = np.delete(theta_set, zero_counts_remove)

        return mu, mu_test, theta_set

    def generate_sample(self, arm):
        genre_idx = self.test_data[:,1] == arm
        meta_user_idx = self.test_data[:,2] == self.theta

        ratings = self.test_data[:, genre_idx, meta_user_idx][:,0]

        if ratings.shape[0] == 0:
            return 0
        else:
            return np.random.choice(ratings)

    def confidence_set_intersection(self, empReward, numPulls):
        # First 5 steps of the algorithm
        T = np.sum(numPulls)
        theta_hat = set()
        isCompetitive = np.zeros(self.numArms)

        #Confidence set construction
        with np.errstate(divide='ignore'): # for first few iterations with 0 numpPulls
            for theta in self.theta_set:
                bound = np.sqrt(2*self.alpha*(self.sigma**2)*np.log(T)/numPulls)
                if all(np.abs(self.mu[theta, :] - empReward) <= bound):
                    theta_hat.add(theta)

        if len(theta_hat) == 0:
            return np.ones(self.numArms)

        # Competitive set
        max_mu = np.max(self.mu, axis=1)
        for arm in range(self.numArms):
            if any(self.mu[:, arm] == max_mu):
                isCompetitive[arm] = 1

        return isCompetitive, theta_hat

    def next_arm_selection(self, rewards):
        return np.random.choice(np.flatnonzero(rewards == rewards.max()))

    def UCB(self, num_iterations, totalRounds):

        avg_regret = np.zero((num_iterations, totalRounds))
        for iteration in range(num_iterations):
            numPulls = np.zeros(numArms)
            empReward = np.zeros(numArms)

            regret = np.zeros(totalRounds)
            for t in range(totalRounds):
                with np.errstate(divide='ignore'):
                    ucb = self.UCBSample(empReward, numPulls)
                next_arm = self.next_arm_selection(ucb)

                #Generate reward, update pulls and empirical reward
                reward_sample = self.generate_sample(next_arm)
                empReward[next_arm] = (empReward[next_arm]*numPulls[next_arm] + reward_sample)/(numPulls[next_arm] + 1)
                numPulls[next_arm] = numPulls[next_arm] + 1

                #Evaluate regret
                regret[t] = self.mu_test[self.theta, bestArm] - mu_test[self.theta, next_arm]

            avg_regret[iteration, :] = regret

        return avg_regret

    def UCBSample(self, empReward, numPulls):
        return empReward + np.sqrt(2*self.alpha*(self.sigma**2)*np.log(np.sum(numPulls))/numPulls)

    def ThompsonSample(self, empiricalMean, numPulls):
        sampleArm = np.random.normal(empiricalMean, np.sqrt(self.sigma**2/numPulls))
        return sampleArm

    def TS(self, num_iterations, totalRounds):

        avg_regret = np.zero((num_iterations, totalRounds))
        for iteration in range(num_iterations):
            numPulls = np.zeros(numArms)
            empReward = np.zeros(numArms)

            regret = np.zeros(totalRounds)
            for t in range(totalRounds):
                #Initialise by pulling each arm once
                if t < numArms:
                    numPulls[t] += 1
                    assert numPulls[t] == 1

                    reward = self.generate_sample(t)
                    empReward[t] = reward

                    regret[t] = mu_test[self.theta, bestArm] - mu_test[self.theta, next_arm]

                    continue

                thompson = ThompsonSample(empReward,numPulls)
                next_arm = self.next_arm_selection(thompson)

                #Generate reward, update pulls and empirical reward
                reward_sample = self.generate_sample( next_arm )
                empReward[next_arm] = (empReward[next_arm]*numPulls[next_arm] + reward_sample)/(numPulls[next_arm] + 1)
                numPulls[next_arm] = numPulls[next_arm] + 1

                #Evaluate regret
                regret[t] = mu_test[self.theta, bestArm] - mu_test[self.theta, next_arm]

            avg_regret[iteration, :] = regret

        return avg_regret


    def UCB_C(self, num_iterations, totalRounds):

        avg_regret = np.zero((num_iterations, totalRounds))
        for iteration in range(num_iterations):
            numPulls = np.zeros(numArms)
            empReward = np.zeros(numArms)

            regret = np.zeros(totalRounds)
            for t in range(totalRounds):
                isCompetitive, _ = self.confidence_set_intersection(empReward, numPulls)
                with np.errstate(divide='ignore'):
                    ucb = self.UCBSample(empReward, numPulls)

                if isCompetitive.sum() == 0:
                    next_arm =  =np.random.randint(0,self.numArms)
                else:
                    self.next_arm_selection(ucb*isCompetitive)

                #Generate reward, update pulls and empirical reward
                reward_sample = self.generate_sample(next_arm)
                empReward[next_arm] = (empReward[next_arm]*numPulls[next_arm] + reward_sample)/(numPulls[next_arm] + 1)
                numPulls[next_arm] = numPulls[next_arm] + 1

                #Evaluate regret
                regret[t] = mu_test[self.theta, bestArm] - mu_test[self.theta, next_arm]

            avg_regret[iteration, :] = regret

        return avg_regret

    def TS_C(self, num_iterations, totalRounds):
        avg_regret = np.zero((num_iterations, totalRounds))
        for iteration in range(num_iterations):
            numPulls = np.zeros(numArms)
            empReward = np.zeros(numArms)

            regret = np.zeros(totalRounds)
            for t in range(totalRounds):
                #Initialise by pulling each arm once
                if t < numArms:
                    numPulls[t] += 1
                    assert numPulls[t] == 1

                    reward = self.generate_sample(t)
                    empReward[t] = reward

                    regret[t] = mu_test[self.theta, bestArm] - mu_test[self.theta, next_arm]

                    continue

                isCompetitive, _ = self.confidence_set_intersection(empReward, numPulls)
                thompson = ThompsonSample(empReward,numPulls)

                if isCompetitive.sum() == 0:
                    next_arm =  =np.random.randint(0,self.numArms)
                else:
                    self.next_arm_selection(thompson*isCompetitive)

                #Generate reward, update pulls and empirical reward
                reward_sample = self.generate_sample(next_arm)
                empReward[next_arm] = (empReward[next_arm]*numPulls[next_arm] + reward_sample)/(numPulls[next_arm] + 1)
                numPulls[next_arm] = numPulls[next_arm] + 1

                #Evaluate regret
                regret[t] = mu_test[self.theta, bestArm] - mu_test[self.theta, next_arm]

            avg_regret[iteration, :] = regret

        return avg_regret


    def UCB_S(self, num_iterations, totalRounds):

        avg_regret = np.zero((num_iterations, totalRounds))
        for iteration in range(num_iterations):
            numPulls = np.zeros(numArms)
            empReward = np.zeros(numArms)

            regret = np.zeros(totalRounds)
            for t in range(totalRounds):
                _, theta_hat = self.confidence_set_intersection(empReward, numPulls)

                if len(theta_hat):
                    next_arm = np.random.randint(0,self.numArms)
                else:
                    idx = np.array(list(theta_hat))
                    supReward = np.max(self.mu[idx, :], axis=0)

                    next_arm = self.next_arm_selection(supReward)

                #Generate reward, update pulls and empirical reward
                reward_sample = self.generate_sample(next_arm)
                empReward[next_arm] = (empReward[next_arm]*numPulls[next_arm] + reward_sample)/(numPulls[next_arm] + 1)
                numPulls[next_arm] = numPulls[next_arm] + 1

                #Evaluate regret
                regret[t] = mu_test[self.theta, bestArm] - mu_test[self.theta, next_arm]

            avg_regret[iteration, :] = regret

        return avg_regret

    def run(self, num_iterations=20,T=5000):

        avg_ucb_regret = self.UCB(num_iterations, T)
        avg_ts_regret = self.TS(num_iterations, T)
        avg_ucbc_regret = self.UCB_C(num_iterations, T)
        avg_tsc_regret = self.TS_C(num_iterations, T)
        avg_ucbs_regret = self.UCB_S(num_iterations, T)

        # mean cumulative regret
        self.plot_av_ucb = np.mean(avg_ucb_regret, axis=0)
        self.plot_av_ts = np.mean(avg_ts_regret, axis=0)
        self.plot_av_ucbc = np.mean(avg_ucbc_regret, axis=0)
        self.plot_av_tsc = np.mean(avg_tsc_regret, axis=0)
        self.plot_av_ucbs = np.mean(avg_ucbs_regret, axis=0)

        # std dev over runs
        self.plot_std_ucb = np.sqrt(np.var(avg_ucb_regret, axis=0))
        self.plot_std_ts = np.sqrt(np.var(avg_ts_regret, axis=0))
        self.plot_std_ucbc = np.sqrt(np.var(avg_ucbc_regret, axis=0))
        self.plot_std_tsc = np.sqrt(np.var(avg_tsc_regret, axis=0))
        self.plot_std_ucbs = np.sqrt(np.var(avg_ucbs_regret, axis=0))

        self.save_data()

    def save_data(self):
        algorithms = ['ucb', 'ts', 'ucbc', 'tsc', 'ucbs']
        pathlib.Path(f'data/plot_arrays/').mkdir(parents=False, exist_ok=True)
        for alg in algorithms:
            np.save(f'data/plot_arrays/plot_av_{alg}',
                    getattr(self, f'plot_av_{alg}'))
            np.save(f'data/plot_arrays/plot_std_{alg}',
                    getattr(self, f'plot_std_{alg}'))

    def plot(self):
        spacing = 250
        # Means
        plt.plot(range(0, 5000)[::spacing], self.plot_av_ucb[::spacing], label='UCB', color='red', marker='+')
        plt.plot(range(0, 5000)[::spacing], self.plot_av_ts[::spacing], label='TS', color='yellow', marker='o')
        plt.plot(range(0, 5000)[::spacing], self.plot_av_ucbc[::spacing], label='UCB-C', color='blue', marker='^')
        plt.plot(range(0, 5000)[::spacing], self.plot_av_tsc[::spacing], label='TS-C', color='black', marker='x')
        plt.plot(range(0, 5000)[::spacing], self.plot_av_ucbs[::spacing], label='UCB-S', color='green', marker='*')

        # Confidence bounds
        plt.fill_between(range(0, 5000)[::spacing], (self.plot_av_ucb + self.plot_std_ucb)[::spacing],
                         (self.plot_av_ucb - self.plot_std_ucb)[::spacing], alpha=0.3, facecolor='r')
        plt.fill_between(range(0, 5000)[::spacing], (self.plot_av_ts + self.plot_std_ts)[::spacing],
                         (self.plot_av_ts - self.plot_std_ts)[::spacing], alpha=0.3, facecolor='y')
        plt.fill_between(range(0, 5000)[::spacing], (self.plot_av_ucbc + self.plot_std_ucbc)[::spacing],
                         (self.plot_av_ucbc - self.plot_std_ucbc)[::spacing], alpha=0.3, facecolor='b')
        plt.fill_between(range(0, 5000)[::spacing], (self.plot_av_tsc + self.plot_std_tsc)[::spacing],
                         (self.plot_av_tsc - self.plot_std_tsc)[::spacing], alpha=0.3, facecolor='k')
         plt.fill_between(range(0, 5000)[::spacing], (self.plot_av_ucbs + self.plot_std_ucbs)[::spacing],
                          (self.plot_av_ucbs - self.plot_std_ucbs)[::spacing], alpha=0.3, facecolor='g')
        # Plot
        plt.legend()
        plt.grid(True, axis='y')
        plt.xlabel('Number of Rounds')
        plt.ylabel('Cumulative Regret')
        # Save
        pathlib.Path('data/plots/').mkdir(parents=False, exist_ok=True)
        plt.savefig(f'data/plots/figure.pdf')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iterations', dest='num_iterations', type=int, default=20,
                        help="Number of iterations of each run")
    parser.add_argument('--T', dest='T', type=int, default=5000, help="Number of rounds")

    return parser.parse_args()

def main(args):
    args = parse_arguments()
    bandit_obj = algs()
    bandit_obj.run(args.num_iterations, args.T)
    bandit_obj.plot()

if __name__ == '__main__':
    main(sys.argv)
