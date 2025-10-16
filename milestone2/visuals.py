import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List


def create_skill_radar(categorized_skills: Dict[str, List[str]]) -> go.Figure:
    """Create a radar chart summarizing counts per major category."""
    # pick a set of core categories to display
    display_categories = [
        'programming_languages', 'web_frameworks', 'databases',
        'ml_ai', 'ml_frameworks', 'cloud_platforms', 'devops_tools'
    ]

    labels = [c.replace('_', ' ').title() for c in display_categories]
    values = [len(categorized_skills.get(c, [])) for c in display_categories]

    # ensure non-zero values for visibility
    if sum(values) == 0:
        values = [0.5 for _ in values]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name='Skill Counts',
        marker=dict(color='teal')
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(1, max(values))])),
        showlegend=False,
        title='Skill Counts by Major Category (Radar)'
    )

    return fig


def create_skill_treemap(categorized_skills: Dict[str, List[str]]) -> go.Figure:
    """Create a treemap that shows categories and their top skills."""
    labels = []
    parents = []
    values = []

    for cat, skills in categorized_skills.items():
        cat_label = cat.replace('_', ' ').title()
        labels.append(cat_label)
        parents.append("")
        values.append(len(skills) if skills else 0.1)

        # add a few sample skills under each category
        for s in skills[:6]:
            labels.append(s)
            parents.append(cat_label)
            values.append(1)

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=values, colorscale='Agsunset')
    ))

    fig.update_layout(title='Skill Treemap (Categories → Sample Skills)')
    return fig


def create_skill_sunburst(categorized_skills: Dict[str, List[str]]) -> go.Figure:
    """Create a sunburst chart (categories -> skills)"""
    labels = []
    parents = []
    values = []

    for cat, skills in categorized_skills.items():
        cat_label = cat.replace('_', ' ').title()
        labels.append(cat_label)
        parents.append("")
        values.append(len(skills) if skills else 0.1)

        for s in skills[:10]:
            labels.append(s)
            parents.append(cat_label)
            values.append(1)

    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues='total',
        marker=dict(colors=values, colorscale='Tealgrn')
    ))

    fig.update_layout(title='Interactive Sunburst: Categories → Skills', height=700)
    return fig
