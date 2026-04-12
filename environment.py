def step(self, action: Action) -> StepResult:
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        if self.state.done:
            return StepResult(
                observation=self._get_observation(),
                reward=0.01,
                done=True,
                info={"message": "Episode already done"}
            )

        reward, info = self._apply_action(action)
        self.state.step_count += 1
        self.state.elapsed_time += 3
        self.state.total_reward += reward

        # Check done conditions
        all_triaged = all(
            p.current_priority != ESIPriority.UNASSIGNED
            for p in self.state.patients
        )
        if self.state.step_count >= self.state.max_steps or all_triaged:
            self.state.done = True
            final_score = self._compute_final_score()
            info["final_score"] = final_score
            info["episode_summary"] = self._episode_summary()

        # Hard clamp reward
        safe_reward = float(max(0.01, min(0.99, round(reward, 4))))

        return StepResult(
            observation=self._get_observation(),
            reward=safe_reward,
            done=self.state.done,
            info=info
        )

    # [Keep all other environment methods exactly the same down to _compute_final_score]
    
    def _compute_final_score(self) -> float:
        try:
            if not getattr(self, "state", None) or not getattr(self.state, "patients", None):
                return 0.01

            scores = []
            for p in self.state.patients:
                if p.current_priority == ESIPriority.UNASSIGNED:
                    scores.append(0.01)
                else:
                    diff = abs(int(p.current_priority.value) - int(p.correct_priority.value))
                    raw = 1.0 - diff * 0.30
                    scores.append(float(max(0.01, min(0.99, raw))))

            if not scores:
                return 0.01
                
            raw_score = sum(scores) / len(scores)
            return float(max(0.01, min(0.99, round(raw_score, 4))))
        except Exception:
            return 0.01
