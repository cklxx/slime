from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from slime.rollout.rm_hub.math_utils import grade_answer_verl


def test_grade_answer_verl_accepts_boxed_answer():
    assert grade_answer_verl("Reasoning steps...\nAnswer: \\boxed{25}", "25")


def test_grade_answer_verl_accepts_plain_answer_line():
    assert grade_answer_verl("Here is the result.\nAnswer: 25", "25")


def test_grade_answer_verl_accepts_unboxed_value():
    assert grade_answer_verl("25", "25")


def test_grade_answer_verl_allows_zero_ground_truth():
    assert grade_answer_verl("Answer: \\boxed{0}", 0)


def test_deepscaler_reward_fallback_without_markers():
    response = "Answer: \\boxed{42}"
    assert get_deepscaler_rule_based_reward(response, "42") == 1
