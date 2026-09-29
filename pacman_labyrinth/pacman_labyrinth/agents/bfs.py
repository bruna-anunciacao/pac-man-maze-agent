from __future__ import annotations

from .search_agent_base import SearchAgentBase
from ..search.algorithms import SearchResult, reconstruct_path


class BFSAgent(SearchAgentBase):
    algorithm_name = "bfs"

    def search(self, problem) -> SearchResult:
        frontier = problem.fifo_frontier()
        frontier.push(problem.start)

        start_key = problem.start.as_tuple()
        parents = {start_key: (None, None)}
        expanded_set: set[tuple[int, int]] = set()

        while frontier:
            cur = frontier.pop()
            cur_key = cur.as_tuple()

            if cur_key in expanded_set:
                continue

            if problem.is_goal(cur):
                actions, path = reconstruct_path(parents, cur)
                return problem.solution(actions, path, True)

            expanded_set.add(cur_key)
            for action, nxt, _step_cost in problem.successors(cur):
                nxt_key = nxt.as_tuple()
                if nxt_key not in expanded_set:
                    parents[nxt_key] = (cur_key, action)
                    frontier.push(nxt)

        return problem.failure()
