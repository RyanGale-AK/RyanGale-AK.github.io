---
title: Deep Q-Learning
katex: true
markup: "mmark"
---

This article chronicles my dive into reinforcement learning and the
training of a deep neural network agent to play Pong on the Atari 2600.
This compiles information from a handful of fantastic teachers including
Ryan Benmalek, Lilian Weng, Richard Sutton, Andrej Karpathy, among
others listed in the references.

Background {#background .unnumbered .unnumbered}
==========

Reinforcement learning is a subset of machine learning involving the
training of an agent to optimally interact with its environment. As the
agent explores its environment it will receive rewards that indicate how
well it is doing at its task. The goal of reinforcement learning is to
create agents that can maximize accumulative reward in complex,
large-scale environments. This often takes the form of a robot
interacting with its environment or a video game bot. The video game
environment has been a popular modality because it offers well defined
rewards and is itself simulated. For my delve into reinforcement
learning (RL), I chose to create a bot that can play Atari Pong. This is
in the spirit of the first deep learning RL agent created by Volodymyr
Minh in 2013. After implementing a Deep Q-Network (DQN)
architecture, I explore more recent improvements suggested in the
literature and implement distributed training.\
Most RL problems can be formulated as a Markov Decision Process (MDP).
An MDP is a directed graph that contains possible states in an
environment with each edge corresponding to actions that an agent can
take. Each action has an associated probability of being taken in
relation to the expected reward that is estimated for that action.

Example of a Markov decision process.
<!-- ![](../../images/mdp.jpg) -->

In the case of Pong, the agent has no information about the game engine,
instead we simply feed it a vector of raw pixel values representing the
current screen. For any given game, the sequence of observation-action
pairs that led to the current screen image encodes a distinct state. Now
as the game progresses the sequence of moves becomes very large,
tracking the entire history is computationally infeasible. That's where
the Markov property comes in to save the day: the future only depends on
the current state. Since the current state contains all of the
statistics necessary for making future decisions, the agent only needs
to know the location of the two paddles and ball.

RL models rely on the Markov property.
<!-- ![](images/agent_environment.png) -->

Now the agent can make an optimal decision according to some rule or
strategy it develops as it makes a series of good or bad moves. In RL,
this strategy is called a policy, $$\pi$$. The policy maps an agent's
state, $$s$$, to an action, $$a$$ and can be deterministic or stochastic.
The agent's goal, of course, is having the optimal policy, but it begins
knowing nothing about how to play Pong. Given an image, the agent must
choose to move up, down, or stay still and every transition has a reward
of $$0$$ except for the end game state where the reward is $$\pm 1$$
depending on whether the agent won the round (although in a game both
players compete to 21 points, the agents reward is reset after each
round). But if the agent begins with no knowledge of playing, how does
it develop a good policy?

A **value function** does exactly what it sounds like, it evaluates how
good a given state or observation-action pair is based on the future
expected reward. In Pong the only nonzero reward is at the end of a
round so the agent's current actions may not reap reward until a
thousand or so frames in the future. To combat this sparsity, we assign
discounted rewards based on how close an action is to the receipt of a
reward. If the paddle agent misses the ball, the most recent action will
be assigned a reward closer to $$-1$$ than moves at the beginning of the
game (which are discounted with an exponentially decaying factor). More
formally, the value, under policy $$\pi$$, is the expected return of being
in state $$s$$: $$V_{\pi}(s)=E_{\pi}[G_t|S_t=s]$$ Where $$\boldsymbol G_t$$,
the **return**, is the sum of discounted rewards gained by future
actions (rewards are allocated upon completion of a game):

$$G_t=R_{t+1}+\gamma R_{t+2}+\dots=\sum\limits_{k=0}^{\infty}\gamma^kR_{t+k+1}$$
The discounting factor $$\gamma\in[0,1]$$ penalizes future rewards.

\centering
![The most recent action prior to a reward is the most relevant.
[@discount]](images/discounted.png){width="75mm"}

\

Q-Learning {#q-learning .unnumbered .unnumbered}
==========

Q-learning is a classic algorithm that broke out in the early days of RL
in 1992. It applies an **action-value** function based on the quality
$$\boldsymbol Q_{\pi}$$ of an observation-action pair:

$$Q_{\pi}(s,a)=E_{\pi}[G_t|S_t=s,A_t=a]$$

The algorithm is as follows:

1.  Beginning in state $$S_t$$ pick action $$A_t=argmax_{a\in A}Q(S_t,a)$$
    using $$\epsilon$$-greedy.

    $$\epsilon$$-greedy is a strategy that chooses to take action
    $$A_t=argmax_{a\in A}Q(S_t,a)$$ with probability $$\epsilon$$, otherwise
    choose a move at random. When the agent first begins it knows
    absolutely nothing, it is given an image and must randomly explore
    the game states in order to get enough experience to know which
    moves are good. This is known as exploration. Once enough moves are
    understood as good and bad, choosing $$A_t=argmax_{a\in A}Q(S_t,a)$$
    allows the model to leverage known moves, this is known as
    exploitation. In the beginning of learning we want the agent to
    explore moves randomly and increase the accuracy of action-value
    estimates. As more experience is gained we want to exploit moves we
    know to produce high reward. However, just because an agent knows a
    few good moves does not mean there are not better moves out there, a
    little randomness will nudge an agent away from sub-par behavior.
    $$\epsilon$$-greedy then strikes a balance between exploration and
    exploitation.

2.  Observe reward $$R_{t+1}$$ and achieve state $$S_{t+1}$$

3.  Update:
    $$Q(S_t,A_t)=Q(S_t,A_t)+\alpha (R_{t+1}+\gamma max_{a\in A}Q(S_{t+1},a)-Q(S_t,A_t))$$

    We make an estimate of the action-value of the next move
    $$Q*=max_{a\in A}Q(S_{t+1},a)$$ in order to estimate the expected
    return of $$A_t$$. In theory, we can think of $$Q*$$ as a giant table
    containing all observation-action pairs that are possible in a game
    of pong. As the agent plays dozens and then hundreds of games, this
    table is slowly filled until finally the best move is known for any
    possible state. Unfortunately this rote memorization, while
    theoretically possible, is not computationally feasible for an
    environment with a massive state-space. This is where a deep neural
    net comes in.

4.  $$t=t+1$$ and repeat until optimal action-value estimates are
    discovered

Deep Q-Network {#deep-q-network .unnumbered .unnumbered}
==============

Twenty-one years after the breakthrough of Q-learning, Google DeepMind
presented the first deep neural network using reinforcement learning,
catching the media's attention for being able to beat human performance
on multiple Atari games[@original]. Because 'memorizing' $$Q*$$ is
infeasible when the state-space is very large, this adaptation uses a
neural network to approximate it. If we use the neural net parameters
$$\theta$$, our new action-value function becomes $$Q(s,a;\theta)$$

Here is an implementation of the DQN with a few improvements:

      def run(self, num_episodes):
            # Load policy weights into target network
            self.q_target.load_state_dict(self.q.state_dict())  
            self.optimizer = optim.Adam(self.q.parameters(), lr=self.learning_rate)

            if self.saved_model:
                # Load pretrained model
                self.q.load_state_dict(torch.load(saved_model)) 

            self.beginLogging()
            env = self.env
            best_episode_score = float('-Inf')
            score = 0.0
            total_frames = 0
            
            # Start first game
            state = get_state(env.reset()) 
            for episode in tqdm(
                        range(self.start_episode,self.start_episode +
                        num_episodes)):
                        
                # anneal 100% to 1% over training (exploration vs exploitation)
                epsilon = self.epsilon_decay(total_frames)
                episode_score = 0
                done = False
                while not done:
                    action = self.q.sample_action(
                        torch.Tensor(state).unsqueeze(0).to(device), epsilon)

                    obs, reward, done, info = env.step(action)

                    next_state = get_state(obs)

                    done_mask = 0.0 if done else 1.0
                    self.memory.put((state,action,reward,next_state,done_mask))
                    state = next_state

                    score += reward
                    episode_score += reward

                    if total_frames > self.training_frame_start:
                        self.train()

                    # Copy policy weights to target
                    if total_frames%self.update_target_interval == 0:
                        self.q_target.load_state_dict(self.q.state_dict())
                    # Save policy weights
                    if total_frames%self.save_interval==0:
                        torch.save(self.q.state_dict(),
                        os.path.join(self.save_location, 'policy_%s.pt' % episode))
                    # Reset environment for the next game
                    if done:
                        state = get_state(env.reset())
                    total_frames += 1

                best_episode_score = max(best_episode_score, episode_score)
                # Print updates every episode
                out = "n_episode : {}, Total Frames : {}, Average Score : {:.1f},
                Episode Score : {:.1f}, Best Score : {:.1f}, n_buffer : {}, eps :
                {:.1f}%".format(
                    episode, total_frames, score/episode, episode_score,
                    best_episode_score, len(self.memory), epsilon*100)
                print(out)
                self.log(out)
                
            # save final model weights
            torch.save(self.q.state_dict(), 
                        os.path.join(self.save_location, 
                        'policy_final.pt'))

Full source code can be found [here](github.com/DataScenic/DQN\)
This implementation utilizes an experience replay buffer, a memory bank
of previous observation-action pairs and their associated rewards. By
updating the network from this buffer, we can learn from previous
actions multiple times (providing stronger assurance of correlation
between a specific observation-action and the reward) and make our data
more independent and identically distributed (a property that provides
stronger convergence guarantees for our function approximator $$Q*$$). The
buffer stores the episode (each round of game play) step
$$e = (state,action,reward,next_state,done_mask)$$. During the update
step, samples are randomly drawn from the replay memory which improves
data efficiency and decorrelates experiences that occur close together
in time. After 10,000 episodes and roughly 500,000 frames of experience
this agent was able to achieve impressive results, winning almost every
match-up 21-0.

\centering
![The agent was able to achieve a very high score per match-up (game to
21) against the default bot](images/DQN_pong.png){width="125mm"}

\centering
\includemedia[width=0.6\linewidth,height=0.6\linewidth,activate=pageopen,
passcontext,
transparent,
addresource=video/pong.mp4,
flashvars={source=video/pong.mp4}
]{\includegraphics[width=0.6\linewidth]{images/Screen Shot 2020-07-24 at 2.23.48 PM.png}}{VPlayer.swf}
\

DQN Variants {#dqn-variants .unnumbered .unnumbered}
============

The standard DQN architecture can inflate the value of an
observation-action pair making it difficult to learn better maneuvers
later in training. In recent years there have been a handful of proposed
improvements to the algorithm, I chose to explore double DQN and
prioritized replay buffer.

In Double DQN (DDQN) we decouple the Q-value estimation with a second
neural net allowing the estimates to be regularized[@DDQN]. More
specifically, one network, $$Q_{\theta}$$, chooses the action while a
second, $$Q_{\theta^-}$$ evaluates that action. The new update step is:
$$Q(S_t,a)=R_{t+1}+\gamma Q(S_{t+1}, max_{a\in A}Q(S_{t+1},a;\theta_t);\theta_t^-)$$

The authors of DDQN show that this decoupling effectively reduces the
tendency of the DQN to overestimate observation-action pairs in
large-scale problems.

A second interesting adaptation is updating the Q network with
experiences that have high importance. Instead of drawing randomly from
the replay buffer, with a prioritized replay buffer we choose the
experience that has the biggest shift in Q-value between the current and
next state (known as the temporal difference error). The authors note
that this allows the agent to \"replay important transitions more
frequently, and therefore learn more efficiently\" [@prioritized].

\centering
![Experimenting with DQN
variants.](images/model_comparison.png){width="125mm"}

In my experiments, I've only noticed a very slight increase in
performance with the DDQN. Because the game is not sufficiently complex,
DQN appears to not drastically overestimate the value of certain moves
(in a game like PacMan DDQN will likely show a performance boost).
Similarly, prioritized replay buffer with DDQN achieved similar
performance with a longer training period.

I found that training the DQN to play Ms Pacman and Space Invaders took
a significant number of rounds and was not able to successfully beat the
game. It appears that the algorithm has low sample efficiency when the
Q-function it needs to approximate is very complex (i.e. Pacman). One
solution could be to increase the model capacity to allow for a more
complex Q-function approximation but this will not scale well. Although
initial attempts with DQN variants didn't show promising results,
meta-learning offers a path forward. In future work I'm interested in
applying Hebbian learning rules to develop a more adaptable DQN. In
their recent paper, \"Meta-Learning through Hebbian Plasticity in Random
Networks,\" Elias Najarro and Sebastian Risi from the University of
Copenhagen showed how discovered Hebbian rules enable an agent to
navigate a dynamical 2D-pixel environment[@hebbian]. These network rules
allow the model to continually self-organize its weights which allow it
to adapt to starkly different learning environments. Stay tuned for
updates on my Github DataScenic.

Conclusion {#conclusion .unnumbered .unnumbered}
==========

DQN is an impressively adaptive RL algorithm that models some aspects of
how we learn to play games ourselves; first exploring new moves with the
risk of defeat but the reward of knowledge, next exploiting known moves
enticed by their anticipated reward. I was initially intrigued by the
algorithm's impressive ability to out-perform a human player on 3 of the
57 original Atari games (Pong, Breakout, and Enduro [@original]). Even
more exciting, while I was working on this project DeepMind published
Agent57, the first algorithm to achieve above the human baseline on all
57 games[@agent57]! This monumental achievement celebrates 7 years of
continuous research attempting to surpass the Atari benchmark, and
ushers in an exciting new chapter of RL where the field extends beyond
video games and into real-world applications. Along this same thread,
Google research has recently proposed the Real-World Reinforcement
Learning Challenge Framework[@real]. This provides researchers with a
new benchmark that simulates the challenges of real-world environments,
including dimensions such as safety, delays, and perturbations. While we
all have been adapting to these environmental challenges since day 1,
robots still struggle. It is our duty to ensure these new friends of
ours are trained and deployed with the highest regard to human safety
and impartiality toward gender, race, creed, wealth, and all other
facets of the human dimension.

\pagebreak
\printbibliography