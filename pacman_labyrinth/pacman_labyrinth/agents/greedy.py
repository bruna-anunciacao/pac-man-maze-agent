from __future__ import annotations

from .search_agent_base import SearchAgentBase
from ..search.algorithms import SearchResult, reconstruct_path


class GreedyBestFirstAgent(SearchAgentBase):
    algorithm_name = "greedy"

    def search(self, problem) -> SearchResult:
        frontier = problem.priority_frontier()
        frontier.push(problem.heuristic(problem.start), problem.start)

        start_key = problem.start.as_tuple()
        parents = {start_key: (None, None)}
        dist = {start_key: 0.0}
        expanded_set: set[tuple[int, int]] = set()

        while frontier:
            _priority, cur = frontier.pop()
            cur_key = cur.as_tuple()

            if cur_key in expanded_set:
                continue
            if problem.is_goal(cur):
                actions, path = reconstruct_path(parents, cur)
                return problem.solution(actions, path, dist[cur_key], True)

            expanded_set.add(cur_key)
            for action, nxt, step_cost in problem.successors(cur):
                nxt_key = nxt.as_tuple()
                if nxt_key in parents:
                    continue
                dist[nxt_key] = dist[cur_key] + step_cost
                parents[nxt_key] = (cur_key, action)
                frontier.push(problem.heuristic(nxt), nxt)

        return problem.failure()
