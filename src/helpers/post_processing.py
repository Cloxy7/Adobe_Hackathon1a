def enforce_document_hierarchy(lines_data):
    """
    Corrects the predicted labels based on logical document structure rules.
    This version aims to maintain correct hierarchy (e.g., H1 -> H2 -> H3)
    and reset behavior after body text.
    """
    if not lines_data:
        return []

    # Track the last *true heading level* encountered. This is NOT reset by body text.
    # It reflects the highest level active before the current line.
    last_active_heading_level = 0 # 0 for body text, 1 for TITLE, 2 for H1, 3 for H2 etc.

    # Track the last *assigned heading level* for sequential demotion if needed.
    # This might be reset by body text if a new section starts.
    last_assigned_heading_level_in_sequence = 0

    has_document_title = False # Flag to ensure only one document title (label 1)

    for i, line in enumerate(lines_data):
        current_predicted_label = line['predicted_label']

        # Rule 1: Demote long text blocks predicted as headings.
        if current_predicted_label > 0 and line['word_count'] > 15:
            current_predicted_label = 0

        # Rule 2: Enforce a single document title (Label 1)
        if current_predicted_label == 1:
            if has_document_title:
                # If a title already exists, demote subsequent detected titles to H1 (label 2)
                current_predicted_label = 2
            else:
                has_document_title = True
                # Update last active heading level to reflect the Title
                last_active_heading_level = current_predicted_label
                last_assigned_heading_level_in_sequence = current_predicted_label
        
        # Rule 3: Enforce logical heading hierarchy for H1, H2, H3 (labels 2, 3, 4)
        elif current_predicted_label > 1: # If it's H1, H2, H3
            # Cannot jump more than one level down (e.g., H1 -> H3 directly is invalid)
            if current_predicted_label > last_active_heading_level + 1:
                current_predicted_label = last_active_heading_level + 1
            
            # If a heading appears after body text, its implied "parent" is the last active heading.
            # If no active heading (last_active_heading_level=0) and it's an H2/H3, promote it to H1.
            # This handles cases where the model correctly identifies it as H2/H3 but misses the parent H1.
            if last_active_heading_level == 0 and current_predicted_label > 2: # If no active parent and it's > H1
                 current_predicted_label = 2 # Promote to H1
            
            # Update active heading level
            last_active_heading_level = current_predicted_label
            last_assigned_heading_level_in_sequence = current_predicted_label

        # Rule 4: If it's body text, the next heading can logically start a new sequence
        # (e.g., an H1 after body text is fine)
        elif current_predicted_label == 0:
            # We don't reset last_active_heading_level to 0 here.
            # Instead, it persists, so if an H3 comes after a body paragraph, but its parent H2 was
            # still "active" from before, it can still be demoted only to that H2 level.
            # Only reset last_assigned_heading_level_in_sequence if you want to force new sections.
            pass


        lines_data[i]['predicted_label'] = current_predicted_label

    return lines_data