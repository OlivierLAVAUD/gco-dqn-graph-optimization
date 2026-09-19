# tests/test_greedy.py
"""Diagnostic : pourquoi Greedy selectionne-t-il si peu d'employes ?"""
from collections import Counter
from src.graph.org_graph import greedy_team
from src.data.generate_synthetic import (
    generate_organization,
    generate_project_requirements,
)

G, employees, all_skills = generate_organization(
    n_employees=200, n_departments=8, n_skills=20, seed=42
)
project_skills = generate_project_requirements(all_skills, 5, seed=42)

team = greedy_team(G, project_skills, max_team_size=15)
covered = set()
for e in team:
    covered |= G.nodes[e].get("skills", set())

print("\n=== RESULTAT GREEDY ===")
print(f"Nb employes selectionnes : {len(team)} -> {team}")
print(f"Skills projet requis     : {sorted(project_skills)}")
print(f"Skills couvertes         : {sorted(covered)}")
print(f"Couverture complete ?    : {covered >= project_skills}")

print("\n=== DETAIL DES EMPLOYES ===")
for e in team:
    d = G.nodes[e]
    print(f"  [{e}] {d['name']} ({d['department']}) : {sorted(d['skills'])}")

print("\n=== STATS ORGANISATION ===")
skills_per_emp = [len(G.nodes[n].get("skills", set())) for n in G.nodes()]
print(f"Nb employes total        : {G.number_of_nodes()}")
print(f"Nb skills total          : {len(all_skills)}")
print(f"Skills/employe moyen     : {sum(skills_per_emp)/len(skills_per_emp):.2f}")
print(f"Skills/employe max       : {max(skills_per_emp)}")
print(f"Employes avec >=4 skills : {sum(1 for x in skills_per_emp if x >= 4)}")

print("\n=== DISTRIBUTION SKILLS/EMPLOYE ===")
dist = Counter(skills_per_emp)
for k in sorted(dist):
    print(f"  {k} skills : {dist[k]} employes")

print("\n=== FREQUENCE DES SKILLS PROJET ===")
for sk in sorted(project_skills):
    count = sum(1 for n in G.nodes() if sk in G.nodes[n].get("skills", set()))
    print(f"  {sk} : present chez {count} employes")