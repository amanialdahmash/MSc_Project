import re

import networkx as nx
import pandas as pd


# Calculate the exact probability
def calculate_probability(G, current_node, target_node):
    if current_node == target_node:
        return 1  # We have reached the target node
    neighbors = list(G.neighbors(current_node))
    if not neighbors:
        return 0  # Dead end
    # Calculate the probability based on the deterministic choices
    probability = 0
    for neighbor in neighbors:
        probability += (1 / len(neighbors)) * calculate_probability(G, neighbor, target_node)
    return probability


def load_tree(tree_file_path):
    # Load the saved tree from the GraphML file
    loaded_tree = nx.read_graphml(tree_file_path)
    return loaded_tree


def save_tree(tree, tree_file_path):
    # Save the tree to a GraphML file
    nx.write_graphml(tree, tree_file_path)
    print(f"Tree saved to {tree_file_path}")


def generate_tree_from_csv_stats(stats_file_path):
    # Create a directed graph (tree) using networkx
    tree = nx.DiGraph()
    df = pd.read_csv(stats_file_path)

    # Iterate through every entry in the selected column
    for row_index, row in df.iterrows():
        output = str(row["cleaned output (choices and results)"]).strip()
        idx = row["id solution reached"]
        prev_node = "OG"
        path = ""
        first_entry = None
        is_gar = False

        for lvl, entry in enumerate(output.split('\n\n')):
            # Define a regex pattern to match the first line
            pattern_first_line = r'^(\w+): (\d+)$'

            # Use re.search to find the first line
            match_first_line = re.search(pattern_first_line, entry, re.MULTILINE)
            type_choice = match_first_line.group(1)
            choice_id = match_first_line.group(2)

            # Skip Guarantee Deadlock completions in tree
            if "GD" in type_choice:
                continue

            # Define a regex pattern to match lines with "number: string" format
            pattern_following_lines = r'(\d+): (.*?)(?=\d+: |$)'

            # Use re.findall to find all lines with this format
            matches_following_lines = re.findall(pattern_following_lines, entry, re.DOTALL | re.MULTILINE)

            # Process the matches
            for number, text in matches_following_lines:
                if number == choice_id:
                    if "G" in type_choice and not is_gar:
                        is_gar = True
                        path = ""  # antecedents[row_index]
                    path += text.strip()
                    if not first_entry:
                        first_entry = text.strip()

                    node_name = f"Id{hash(path)}\n{type_choice}:{text.strip()}"
                    tree.add_edge(prev_node, node_name)
                    prev_node = node_name
                    break
        tree.add_edge(prev_node, str(idx))

    return tree


if __name__ == '__main__':
    stats_file = "out/stats_remote_minepump_v1.csv"
    tree = generate_tree_from_csv_stats(stats_file)
    print(calculate_probability(tree, current_node="OG", target_node="0"))
    # Specify the file path to save the tree
    # tree_file_path = 'tree.graphml'
    # save_tree(tree, tree_file_path)

    # Automate node renaming
    node_mapping = {}
    for node in tree.nodes:
        if not node.isdigit():
            unique_id = id(node)
            prob = calculate_probability(tree, current_node=node, target_node='0')
            last_6_digits = str(unique_id)[-6:]
            ch_type = node.split(':')[0].split('\n')[-1]
            new_name = f"{last_6_digits}\n{ch_type}: {prob:.3f}".rstrip('0').rstrip('.')

        else:
            new_name = node
        node_mapping[node] = new_name

    # Rename nodes in the graph
    tree = nx.relabel_nodes(tree, node_mapping)

    # Convert NetworkX graph to Graphviz Digraph
    A = nx.nx_agraph.to_agraph(tree)
    A.node_attr.update(fontsize=24)

    target_node = A.get_node('0')
    target_node.attr['fontname'] = 'bold'
    target_node.attr['penwidth'] = '5'

    # Render the Graphviz AGraph to an image file using Graphviz
    output_file = "visualisations/remote/tree.png"  # Replace with your desired output file name
    A.draw(output_file, format='png', prog='dot')
