import numpy as np

def viterbi(model, observation, tags):
  v =[] # viterbi matrix
  for i in range(len(observation)):
    v.append([])
    for j in range(len(tags)):
      v[i].append(0.0)
  vb = [] # backpointer
  for i in range(len(observation)):
    vb.append([])
    for j in range(len(tags)):
      vb[i].append("")

  for i in range(0, len(tags)):
    v[0][i] = model.get_trellis_arc(tags[i], None, observation, 0)
  
  # Dynamic programming step
  for i in range(1,len(observation)):
    for j in range(0, len(tags)):
      maximum_score = -np.inf
      max_prob_tag = ""
      for k in range(0, len(tags)):
        prob = model.get_trellis_arc(tags[j], tags[k], observation, i)
        score = v[i-1][k] + prob

        # Don't favor 'O' tags
        if tags[j] == "O":
           score -= 0.6
           if tags[k] == "O":
             score -= 1
        if tags[j].split("-")[-1] == "MISC":
          score -= 0.5
        if tags[j].split("-")[-1] == "ORG":
          score -= 0.2

        # Repetition penalty
        if tags[j] == tags[k]:
          score -= 0.1
        if len(tags[j].split("-")) == 2 and len(tags[k].split("-")) == 2:
          if tags[j].split("-")[1] == tags[k].split("-")[1]:
            score -= 0.3

        # Capitalization usually means it's a PER or LOC
          if tags[j].split("-")[1] == "PER" or tags[j].split("-")[1] == "LOC":
            if observation[i][0].isupper():
              score += 0.5
            else:
              score -= 0.05

        if score > maximum_score:
          maximum_score = score
          max_prob_tag = tags[k]
      v[i][j] = maximum_score
      vb[i][j] = max_prob_tag

  #Termination
  best = [None] * len(observation)
  maximum_score = -np.inf


  for i in range(0, len(tags)):
    # Start with last tag
    score = v[len(observation) -1][i]
    if score > maximum_score:
      maximum_score = score
      best[len(observation) -1] = tags[i]

  for i in range(len(observation) - 2, -1, -1):
    # Backtrack
    best[i] = vb[i + 1][tags.index(best[i + 1])]

  return best
