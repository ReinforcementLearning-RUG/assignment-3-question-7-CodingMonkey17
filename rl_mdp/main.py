import numpy as np
from rl_mdp.mdp.abstract_mdp import AbstractMDP
from rl_mdp.policy.abstract_policy import AbstractPolicy
from rl_mdp.policy.policy import Policy
from rl_mdp.model_free_prediction.td_evaluator import TDEvaluator
from rl_mdp.model_free_prediction.td_lambda_evaluator import TDLambdaEvaluator
from rl_mdp.model_free_prediction.monte_carlo_evaluator import MCEvaluator  # Assuming you have an MC evaluator
from rl_mdp.util import create_mdp, create_policy_1, create_policy_2


def main():
    # Step 1: Create the MDP
    mdp = create_mdp()

    # Step 2: Create policies π1 and π2
    policy_1 = create_policy_1()
    policy_2 = create_policy_2()

    # Parameters for all evaluators
    alpha = 0.1  # Step size
    num_episodes = 1000  # Number of episodes to run for evaluation

    # Step 3: Initialize TD(0), TD(λ), and Monte Carlo evaluators for both policies
    td_evaluator_pi1 = TDEvaluator(mdp, alpha)
    td_evaluator_pi2 = TDEvaluator(mdp, alpha)

    td_lambda_evaluator_pi1 = TDLambdaEvaluator(mdp, alpha, lambd=0.5)
    td_lambda_evaluator_pi2 = TDLambdaEvaluator(mdp, alpha, lambd=0.5)

    mc_evaluator_pi1 = MCEvaluator(mdp)
    mc_evaluator_pi2 = MCEvaluator(mdp)

    # Step 4: Evaluate the value functions for π1 and π2 using all three methods

    V_pi1_td = td_evaluator_pi1.evaluate(policy_1, num_episodes)
    V_pi1_td_lambda = td_lambda_evaluator_pi1.evaluate(policy_1, num_episodes)
    V_pi1_mc = mc_evaluator_pi1.evaluate(policy_1, num_episodes)

    V_pi2_td = td_evaluator_pi2.evaluate(policy_2, num_episodes)
    V_pi2_td_lambda = td_lambda_evaluator_pi2.evaluate(policy_2, num_episodes)
    V_pi2_mc = mc_evaluator_pi2.evaluate(policy_2, num_episodes)

    # Step 5: Display the results for Policy π1
    print("Estimated Value Function for π1 (TD(0)):")
    print(V_pi1_td)
    print("Estimated Value Function for π1 (TD(λ)):")
    print(V_pi1_td_lambda)
    print("Estimated Value Function for π1 (Monte Carlo):")
    print(V_pi1_mc)

    # Step 6: Display the results for Policy π2
    print("\nEstimated Value Function for π2 (TD(0)):")
    print(V_pi2_td)
    print("Estimated Value Function for π2 (TD(λ)):")
    print(V_pi2_td_lambda)
    print("Estimated Value Function for π2 (Monte Carlo):")
    print(V_pi2_mc)

    # Step 7: Compare the policies excluding the last element
    print("\nComparison of Value Functions for Policies π1 and π2:")

    # TD(0) Comparison
    print("------- TD(0) -------")
    if np.all(V_pi1_td[:-1] > V_pi2_td[:-1]):
        print("Policy π1 is better than π2 (TD(0)).")
    else:
        print("Policy π1 is not strictly better than π2 (TD(0)), but let's compare values state by state.")
    for s in range(mdp.num_states - 1):  # Exclude the last state
        print(f"State {s} (TD(0)): V_pi1 = {V_pi1_td[s]}, V_pi2 = {V_pi2_td[s]}")

    # TD(λ) Comparison
    print("------- TD(λ) -------")
    if np.all(V_pi1_td_lambda[:-1] > V_pi2_td_lambda[:-1]):
        print("Policy π1 is better than π2 (TD(λ)).")
    else:
        print("Policy π1 is not strictly better than π2 (TD(λ)), but let's compare values state by state.")
    for s in range(mdp.num_states - 1):  # Exclude the last state
        print(f"State {s} (TD(λ)): V_pi1 = {V_pi1_td_lambda[s]}, V_pi2 = {V_pi2_td_lambda[s]}")

    # Monte Carlo Comparison
    print("------- Monte Carlo -------")
    if np.all(V_pi1_mc[:-1] > V_pi2_mc[:-1]):
        print("Policy π1 is better than π2 (Monte Carlo).")
    else:
        print("Policy π1 is not strictly better than π2 (Monte Carlo), but let's compare values state by state.")
    for s in range(mdp.num_states - 1):  # Exclude the last state
        print(f"State {s} (Monte Carlo)): V_pi1 = {V_pi1_mc[s]}, V_pi2 = {V_pi2_mc[s]}")

if __name__ == "__main__":
    main()
