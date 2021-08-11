import numpy as np

import pymc3 as pm
import theano.tensor as tt


def estimate_latent_ability_membership(observed_scores, number_questions,
                                       beta_1_kwargs, beta_2_kwargs,
                                       bernoulli_kwargs):
    """PyMC3 implementation of latent ability and membership model.

    Args:
        - observed_scores: numpy array, observed correct answers for each
            instance.
        - number_questions: int, number of total questions.
        - beta_1_kwargs: dict, keyword arguments for a beta distrbution.
        - beta_2_kwargs: dict, keyword arguments for a beta distrbution.
        - bernoulli_kwargs: dict,  keyword arguments for a bernoulli
            distrbution.

    Returns:
        - model: is a PyMC3 model, it simultaneously estimates the membership
            to two groups and the latent ability associated with this two
            groups.

    """
    with pm.Model() as model:

        latent_group_1_ability = pm.Beta(
            'latent_group_1_ability',
            **beta_1_kwargs
        )
        latent_group_2_ability = pm.Beta(
            'latent_group_2_ability',
            **beta_2_kwargs
        )

        groups_abilities = [latent_group_1_ability, latent_group_2_ability]

        latent_group_membership = pm.Bernoulli(
            'latent_group_membership',
            shape=(len(observed_scores),),
            **bernoulli_kwargs
        )

        observed_scores = pm.Binomial(
            'observed_scores',
            n=number_questions,
            p=groups_abilities[latent_group_membership],
            observed=observed_scores
        )

    return model


def estimate_latent_ability_hier(observed_scores, number_questions,
                                 a_beta_1_kwargs, b_beta_1_kwargs,
                                 a_beta_2_kwargs, b_beta_2_kwargs,
                                 bernoulli_kwargs):
    """PyMC3 implementation of latent ability and membership model with
    hierarchical structure.

    Args:
        - observed_scores: numpy array, observed correct answers for each
            instance.
        - number_questions: int, number of total questions.
        - a_beta_1_kwargs: dict, keyword arguments for an uniform
            distrbution. It parametrizes the a parameter of a beta
            distribution.
        - b_beta_1_kwargs: dict, keyword arguments for an uniform
            distrbution. It parametrizes the b parameter of a beta
            distribution
        - bernoulli_kwargs: dict,  keyword arguments for a bernoulli
            distrbution.

    Returns:
        - model: is a PyMC3 model, it simultaneously estimates the
            membership to two groups and the latent ability associated with
            this two groups.
    """
    with pm.Model() as model:

        idx = pm.Data(
            'observation_indices',
            np.array(
                [i for i in range(len(observed_scores))]
            )
        )

        hyper_a_beta_1 = pm.Uniform(
            'hyper_a_beta_1',
            **a_beta_1_kwargs
        )
        hyper_b_beta_1 = pm.Uniform(
            'hyper_b_beta_1',
            **b_beta_1_kwargs
        )
        latent_group_1_ability = pm.Beta(
            'latent_group_1_ability',
            a=hyper_a_beta_1,
            b=hyper_b_beta_1,
            shape=(len(observed_scores),)
        )

        hyper_a_beta_2 = pm.Uniform(
            'hyper_a_beta_2',
            **a_beta_2_kwargs
        )
        hyper_b_beta_2 = pm.Uniform(
            'hyper_b_beta_2',
            **b_beta_2_kwargs
        )
        latent_group_2_ability = pm.Beta(
            'latent_group_2_ability',
            a=hyper_a_beta_2,
            b=hyper_b_beta_2,
            shape=(len(observed_scores),)
        )

        groups_abilities = [latent_group_1_ability, latent_group_2_ability]

        latent_group_membership = pm.Bernoulli(
            'latent_group_membership',
            shape=(len(observed_scores),),
            **bernoulli_kwargs
        )

        observed_scores = pm.Binomial(
            'observed_scores',
            n=number_questions,
            p=groups_abilities[latent_group_membership][idx],
            observed=observed_scores
        )

    return model


def estimate_difficulty_ability(observed_answers, question_idx, answer_idx,
                                difficulty_beta_kwargs, ability_beta_kwargs):
    """PyMC3 model for estimating the latent difficulty of a question and
    ability of who answered. The observed_answers come in format of pairs of
    questions / answers_id.

    Args:
        - observed_answers: array of binary, weather if the pair of question
            answer is correct / wrong.
        - question_idx: array of int, question indices associated to the
            answers.
        - answer_idx: array of int, anwsers indices associated to the
            answers.
        - difficulty_beta_kwargs: dict, keyword arguments for a beta
            distribution.
        - ability_beta_kwargs: dict, keyword arguments for a beta distribution.

    Returns:
        - model: a PyMC3 model, it estimates the joint probability of an answer
            being correct given the latent difficulty of that question and
            the latent ability of who answered to the question.
    """
    with pm.Model() as model:

        question_idx = pm.Data(
            'question_idx',
            question_idx
        )
        answer_idx = pm.Data(
            'answer_idx',
            answer_idx
        )

        difficulty_questions = pm.Beta(
            'difficulty_questions',
            shape=(len(question_idx.unique())),
            **difficulty_beta_kwargs
        )
        ability_answers = pm.Beta(
            'ability_answers',
            shape=(len(answer_idx.unique())),
            **ability_beta_kwargs
        )

        p_answer_correct = pm.Deterministic(
            'p_answer_correct',
            difficulty_questions[question_idx] * ability_answers[answer_idx]
        )

        correct_answers = pm.Bernoulli(
            'observed_correct_answers',
            p=p_answer_correct,
            observed=observed_answers
        )

    return model


def estimate_conditional_difficulty_ability(observed_answers, question_idx,
                                            answer_idx, uniform_kwargs,
                                            bernoulli_question_kwargs,
                                            bernoulli_answer_kwargs
                                            ):
    """PyMC3 model for estimating the latent difficulty of a question and
    ability of who answered conditioned on the membership to a group.
    The observed_answers come in format of pairs of questions / answers.

    Args:
        - observed_answers: array of binary, weather if the pair of question
            answer is correct / wrong.
        - question_idx: array of int, question indices associated to the
            answers.
        - answer_idx: array of int, anwsers indices associated to the
            answers.
        - uniform_kwargs: dict, keyword arguments for an unifiorm
            distribution.
        - bernoulli_question_kwargs: dict, keyword arguments for a bernoulli
            distribution.
        - bernoulli_answer_kwargs: dict, keyword arguments for a bernoulli
            distribution.

    Returns:
        - model: a PyMC3 model, it estimates the joint probability of an answer
            being correct given the latent difficulty of that question and
            the latent ability of who answered to the question.
    """
    with pm.Model() as model:

        question_idx = pm.Data(
            'question_idx',
            question_idx
        )
        answer_idx = pm.Data(
            'answer_idx',
            answer_idx
        )

        latent_group_membership_question = pm.Bernoulli(
            'latent_group_membership_question',
            shape=(len(question_idx.unique()), ),
            **bernoulli_question_kwargs
        )
        latent_group_membership_answer = pm.Bernoulli(
            'latent_group_membership_answer',
            shape=(len(answer_idx.unique()), ),
            **bernoulli_answer_kwargs
        )

        same_group_latent_ability = pm.Uniform(
            'same_group_latent_ability',
            **uniform_kwargs
        )
        different_group_latent_ability = pm.Uniform(
            'different_group_latent_ability',
            lower=0,
            upper=same_group_latent_ability
        )

        latent_abilities = [
            same_group_latent_ability, different_group_latent_ability
        ]
        ability_idx = latent_group_membership_question[question_idx] \
            == latent_group_membership_answer[answer_idx]
        ability_idx = ability_idx * 1

        correct_answers = pm.Bernoulli(
            'observed_correct_answers',
            p=latent_abilities[ability_idx],
            observed=observed_answers
        )

    return model


def estimate_malingering_hier(observed_answers, n_questions,
                              beta_mu_b_kwargs, half_norm_mu_delta_kwargs,
                              uniform_lamba_kwargs, uniform_lambda_malinger,
                              beta_phi_kwargs):
    """
    """
    with pm.Model() as model:

        mu_b = pm.Beta(
            'mu_benevolent',
            **beta_mu_b_kwargs
        )
        mu_delta = pm.HalfNormal(
            'mu_delta',
            **half_norm_mu_delta_kwargs
        )
        lam_b = pm.Uniform(
            'lambda_benevolent',
            **uniform_lamba_kwargs
        )
        lam_m = pm.Uniform(
            'lambda_malinger',
            **uniform_lambda_malinger
        )
        phi = pm.Beta(
            'phi',
            **beta_phi_kwargs
        )
        membership = pm.Bernoulli(
            'latent_group_membership',
            p=phi,
            shape=len(observed_answers)
        )
        mu_m = pm.Deterministic(
            'mu_malinger',
            1 / (1 + tt.exp(tt.log(1 / mu_b - 1) + mu_delta))
        )

        p_b = pm.Beta(
            'probability_benevolent',
            alpha=mu_b*lam_b,
            beta=(1 - mu_b) / lam_b
        )
        p_m = pm.Beta(
            'probability_malevolent',
            alpha=mu_m*lam_m,
            beta=(1 - mu_m) / lam_m
        )

        ps = [p_b, p_m]

        answers = pm.Binomial(
            'observed_answers',
            n=n_questions,
            p=ps[membership],
            observed=observed_answers
        )

    return model
