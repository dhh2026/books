import re
import string

def clean_text(text):
    """
    Converts text to lowercase, removes all punctuation, 
    and strips leading/trailing whitespace.
    """
    if text is None:
        return ''
        
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove all punctuation using a translation table
    punctuation_remover = str.maketrans('', '', string.punctuation)
    text = text.translate(punctuation_remover)
    
    # 3. Strip leading/trailing whitespace
    return text.strip()


def clean_author(text):
    # Extract all key-value pairs in their exact sequential order
    pairs = dict(re.findall(r'([^|$]+)\$([^|]*)', text.lower()))

    author = ''
    # authors = []
    # current_record = {}
    
    for key, value in pairs.items():

        if key in ('7', 'a', 'd', 'p'):
            author_name = ''.join([clean_text(pairs.get('p', '')).replace(' ', ''), 
                        clean_text(pairs.get('a', '')).replace(' ', ''), 
                        clean_text(pairs.get('d', '')).replace(' ', '')])
            author = ', '.join([pairs.get('7', ''), 
                                author_name])
        
    #     if key_lower in ('7', 'a', 'd', 'p'):
    #         # If we see a tag we already have, package the current record and reset
    #         if key_lower in current_record:
    #             author_name = ''.join([clean_text(current_record.get('p', '')).replace(' ', ''), 
    #                                    clean_text(current_record.get('a', '')).replace(' ', ''), 
    #                                    clean_text(current_record.get('d', '')).replace(' ', '')])
    #             author = ', '.join([current_record.get('7', ''), 
    #                                 author_name])
    #             authors.append(author)
    #             current_record = {}
            
    #         current_record[key_lower] = value
            
    # # Save the last record remaining after the loop finishes
    # if current_record:
    #     author_name = ''.join([clean_text(current_record.get('p', '')).replace(' ', ''), 
    #                             clean_text(current_record.get('a', '')).replace(' ', ''), 
    #                             clean_text(current_record.get('d', '')).replace(' ', '')])
    #     author = ', '.join([current_record.get('7', ''), 
    #                         author_name])
    #     authors.append(author)
    
    return author